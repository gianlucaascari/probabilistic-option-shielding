import re
import gc
import time
import copy
import torch
import logging
import importlib
import numpy as np
from z3 import *
from ...core.config import *
from multiprocessing import Pool
from ...utils.find import find_agent
from ...utils.sample import split_into_subsets
from .z3 import Z3Planner
from .z3 import PesudoAgent

logger = logging.getLogger(__name__)

# Module-level cache for grounding functions (avoids importlib.import_module per step)
_grounding_fn_cache = {}

def _get_grounding_fn(method_full_name):
    """Cache importlib.import_module + getattr results."""
    if method_full_name not in _grounding_fn_cache:
        module_name, method_name = method_full_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        _grounding_fn_cache[method_full_name] = getattr(module, method_name)
    return _grounding_fn_cache[method_full_name]


class Z3PlannerRL(Z3Planner):
    _EVAL_CACHE_MAX = 200000

    def __init__(self, yaml_path):
        super().__init__(yaml_path)
        self.rl_input_shape = None
        self.last_rl_obs = None
        # Skip the per-agent Z3 solve when the Sim ruleset is empty (see
        # _default_sim_action_dist). Set to False to force the legacy path.
        self.fast_empty_sim = True
        self._default_action_dist_cache = {}

    def reset(self):
        self.last_rl_obs = None

    def _default_sim_action_dist(self, agent):
        """The action distribution a Sim solve degenerates to with no Sim rules.

        With an empty Sim ruleset, solve_sub_problem's solver only ever receives
        the grounding facts plus the closed-world constraints, so it is always
        sat. The action predicates (Stop/Fast/Normal/Slow) have function "None"
        and are therefore never asserted, leaving them unconstrained, so every
        is_true(model.evaluate(...)) is False and action_dist stays all-zero.
        The solve then always falls through to its "Normal" default, which is
        what we return directly here.
        """
        key = id(agent.action_mapping)
        cached = self._default_action_dist_cache.get(key)
        if cached is None:
            cached = torch.zeros_like(agent.action_dist)
            for action_id, action_name in agent.action_mapping.items():
                if "Normal" in action_name:
                    cached[action_id] = 1.0
            self._default_action_dist_cache[key] = cached
        # Callers treat the distribution as their own, so hand out a copy.
        return cached.clone()

    def _create_rules(self):
        assert "Task" in self.data["Rules"].keys(), "Make sure the task rule is defined"
        assert "Sim" in self.data["Rules"].keys(), "Make sure the sim rule is defined"
        self.z3_vars = []
        self.rules = {
            "Task": {},
            "Sim": {}
        }
        for rule_dict in self.data["Rules"]["Sim"] or []:
            rule_name, formula = rule_dict["name"], rule_dict["formula"]
            # Check if the rule is valid
            logger.info("*** Sim Rule ***: {} -> \n {}".format(rule_name, formula))

            # Create Z3 variables based on the formula
            var_names = self._extract_variables(formula)
            # Note: Sim rule should contain all the Z3 variables

            # Substitute predicate names in the formula with Z3 function instances
            for method_name, pred_info in self.predicates.items():
                formula = formula.replace(method_name, f'local_predicates["{method_name}"]["instance"]')

            # Now replace the variable names in the formula with their Z3 counterparts
            for var_name in var_names:
                formula = formula.replace(var_name, f'z3_vars["{var_name}"]')
                if var_name not in self.z3_vars:
                    self.z3_vars.append(var_name)

            # Evaluate the modified formula string to create the Z3 expression
            self.rules["Sim"][rule_name] = formula

        for rule_dict in self.data["Rules"]["Task"]:
            rule_name, formula = rule_dict["name"], rule_dict["formula"]
            self.rules["Task"][rule_name] = {}
            # Check if the rule is valid
            logger.info("*** Task Rule ***: {} -> \n {}".format(rule_name, formula))

            # Create Z3 variables based on the formula
            var_names = self._extract_variables(formula)

            # Substitute predicate names in the formula with Z3 function instances
            for method_name, pred_info in self.predicates.items():
                formula = formula.replace(method_name, f'local_predicates["{method_name}"]["instance"]')

            # Now replace the variable names in the formula with their Z3 counterparts
            for var_name in var_names:
                formula = formula.replace(var_name, f'z3_vars["{var_name}"]')

            # Evaluate the modified formula string to create the Z3 expression
            self.rules["Task"][rule_name]["content"] = formula
            self.rules["Task"][rule_name]["reward"] = rule_dict["reward"]
            self.rules["Task"][rule_name]["immediate"] = rule_dict["immediate"] if "immediate" in rule_dict else False
            self.rules["Task"][rule_name]["dead"] = rule_dict["dead"]
        
        logger.info("Rules created successfully")
        logger.info("Rules will be grounded later...")

    def plan(self, world_matrix, 
             intersect_matrix, 
             direction_matrix,
             agents, 
             layerid2listid, 
             use_multiprocessing=True, 
             rl_agent=None):
        # 1. Break the global world matrix into local world matrix and split the agents and intersections
        # Note that the local ones will have different size and agent id
        # With no Sim rules every non-RL solve returns the same constant, so those
        # agents need neither a solve nor a field-of-view crop. This dominates step
        # time once pedestrians are active.
        skip_sim = self.fast_empty_sim and not self.rules['Sim']

        local_world_matrix = world_matrix.clone()
        local_intersections = intersect_matrix.clone()
        local_directions = direction_matrix.clone()
        ego_agent, partial_agents, partial_world, partial_intersections, partial_directions, rl_flags = \
            self.break_world_matrix(local_world_matrix, agents, local_intersections, local_directions,
                                    layerid2listid, rl_agent, rl_only=skip_sim)
        # 2. Choose between multi-processing and looping
        combined_results = {}
        agent_keys = list(partial_agents.keys())

        if skip_sim:
            # partial_agents only holds the RL agent now, so fill in the rest here.
            for ego_name, agent in ego_agent.items():
                if not rl_flags[ego_name]:
                    combined_results[ego_name] = self._default_sim_action_dist(agent)

        if use_multiprocessing and agent_keys:
            # Multi-processing approach
            agent_batches = split_into_batches(agent_keys, NUM_PROCESS)
            with Pool(processes=NUM_PROCESS) as pool:
                for batch_keys in agent_batches:
                    batch_results = pool.starmap(solve_sub_problem, 
                                                [(ego_name, ego_agent[ego_name].action_mapping, ego_agent[ego_name].action_dist,
                                                self.rule_tem, self.entity_types, self.predicates, self.z3_vars,
                                                partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name],
                                                partial_directions[ego_name], self.fov_entities, rl_flags[ego_name], 
                                                self.data['Constants'] if 'Constants' in self.data else None, 
                                                self.rl_input_shape)
                                                for ego_name in batch_keys])
                    
                    for result in batch_results:
                        combined_results.update(result)
                    gc.collect()
        else:
            # Looping approach
            for ego_name in agent_keys:
                if rl_flags[ego_name]:
                    # RL agent only gets the observation - use fast path without Z3
                    if hasattr(self, '_pred_domain_types'):
                        result = self._solve_grounding_fast(
                            ego_name, partial_agents[ego_name],
                            partial_world[ego_name], partial_intersections[ego_name],
                            partial_directions[ego_name])
                    else:
                        result = solve_sub_problem(ego_name, ego_agent[ego_name].action_mapping, ego_agent[ego_name].action_dist,
                                                self.rules['Task'], self.entity_types, self.predicates, self.z3_vars,
                                                partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name], 
                                                partial_directions[ego_name], self.fov_entities, True, 
                                                constants=self.data['Constants'] if 'Constants' in self.data else None, 
                                                rl_input_shape=self.rl_input_shape)
                    self.last_rl_obs = {
                        "last_obs_dict": result["{}_grounding_dic".format(ego_name)],
                        "last_obs": result["{}_grounding".format(ego_name)].copy()
                    }
                else:
                    result = solve_sub_problem(ego_name, ego_agent[ego_name].action_mapping, ego_agent[ego_name].action_dist,
                                            self.rules['Sim'], self.entity_types, self.predicates, self.z3_vars,
                                            partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name], 
                                            partial_directions[ego_name], self.fov_entities, False, 
                                            constants=self.data['Constants'] if 'Constants' in self.data else None,)
                combined_results.update(result)

        return combined_results
    
    def eval(self, rl_action, immediate=False):
        if self.last_rl_obs is None:
            return False, 0, []
        # Use cached fast path if available
        if hasattr(self, '_eval_solver'):
            fast_result = self._eval_fast(rl_action, immediate)
            return fast_result
        # Fallback to original implementation
        fail, reward, violations = eval_action(rl_action, self.rules['Task'], self.entity_types, self.predicates, self.z3_vars, self.fov_entities,
                             self.last_rl_obs["last_obs_dict"], self.last_rl_obs["last_obs"], self.data['Constants'] if 'Constants' in self.data else None,
                             immediate)
        return fail, reward, violations

    def break_world_matrix(self, world_matrix, agents, intersect_matrix, direction_matrix, layerid2listid, rl_agent, rl_only=False):
        """Crop the world into per-agent field-of-view views.

        With rl_only=True only the RL agent gets a view: the caller has
        determined no other agent will be handed to the solver, and cropping
        plus cloning three matrices per agent is the bulk of the cost here.
        """
        ego_agent = {}
        partial_agents = {}
        partial_world = {}
        partial_intersection = {}
        partial_direction = {}
        rl_flag = {}
        for agent in agents:
            ego_name = "{}_{}".format(agent.type, agent.layer_id)
            ego_agent[ego_name] = agent
            rl_flag[ego_name] = (agent.layer_id==rl_agent)
            if rl_only and not rl_flag[ego_name]:
                continue
            ego_layer = world_matrix[agent.layer_id]
            assert len((ego_layer == TYPE_MAP[agent.type]).nonzero()) == 1, ValueError("Ego agent {}_{} should be unique in the world matrix, now it is {}".format(agent.type, agent.layer_id, (ego_layer == TYPE_MAP[agent.type]).nonzero()))
            ego_position = (ego_layer == TYPE_MAP[agent.type]).nonzero()[0]
            ego_direction = agent.last_move_dir
            x_start, y_start, x_end, y_end = self.get_fov(ego_position, ego_direction, world_matrix.shape[1], world_matrix.shape[2])
            partial_world_all = world_matrix[:, x_start:x_end, y_start:y_end].clone()
            partial_intersections = intersect_matrix[:, x_start:x_end, y_start:y_end].clone()
            partial_directions = direction_matrix[:, x_start:x_end, y_start:y_end].clone()
            partial_world_nonzero_int = torch.logical_and(partial_world_all != 0, \
                                                          partial_world_all == partial_world_all.to(torch.int64))
            # Apply torch.any across dimensions 1 and 2 sequentially
            non_zero_layers = partial_world_nonzero_int.any(dim=1).any(dim=1)
            non_zero_layer_indices = torch.where(non_zero_layers)[0]
            partial_world_squeezed = partial_world_all[non_zero_layers]
            partial_agent = {}
            for layer_id in range(partial_world_squeezed.shape[0]):
                layer = partial_world_squeezed[layer_id]
                layer_nonzero_int = torch.logical_and(layer != 0, layer == layer.to(torch.int64))
                if layer_nonzero_int.nonzero().shape[0] > 1:
                    continue
                if len(partial_agent) >= self.fov_entities["Entity"] and rl_flag[ego_name]:
                    # can only handle fixed number of agents
                    break
                non_zero_values = int(layer[layer_nonzero_int.nonzero()[0][0], layer_nonzero_int.nonzero()[0][1]])
                agent_type = LABEL_MAP[non_zero_values]
                # find this agent
                other_agent_layer_id = int(non_zero_layer_indices[layer_id])
                other_agent = agents[layerid2listid[other_agent_layer_id]]
                assert other_agent.type == agent_type
                if other_agent_layer_id == agent.layer_id:
                    partial_agent["ego_{}".format(layer_id)] = PesudoAgent(agent_type, layer_id, other_agent.concepts, other_agent.last_move_dir, other_agent.pos)
                else:
                    partial_agent[str(layer_id)] = PesudoAgent(agent_type, layer_id, other_agent.concepts, other_agent.last_move_dir, other_agent.pos)
            if rl_flag[ego_name]:
                # RL agent needs fixed number of entities
                ph_concepts = {
                    "type": "PH",
                    "priority": 0,
                }
                while len(partial_agent) < self.fov_entities["Entity"]:
                    layer_id += 1
                    place_holder_agent = PesudoAgent("PH", layer_id, ph_concepts, \
                                                      None)
                    partial_agent["PH_{}".format(layer_id)] = place_holder_agent
            partial_world[ego_name] = partial_world_squeezed
            partial_intersection[ego_name] = partial_intersections
            partial_direction[ego_name] = partial_directions
            partial_agents[ego_name] = partial_agent
        return ego_agent, partial_agents, partial_world, partial_intersection, partial_direction, rl_flag
            
    def logic_grounding_shape(self, fov_entities):
        self.fov_entities = fov_entities
        self.rl_input_shape, pred_grounding_ind = logic_grounding_shape(self.entity_types, self.predicates, self.z3_vars, fov_entities)
        self.pred_grounding_index = pred_grounding_ind
        self._init_eval_cache()
        return self.rl_input_shape + 3, pred_grounding_ind  # + [row, col, school_time]

    def _init_eval_cache(self):
        """Pre-compute Z3 objects to avoid recreating them every step."""
        # Build a namespace dict for eval() calls
        ns = {}

        # Create Z3 sorts ONCE
        entity_sorts = {}
        for et in self.entity_types:
            entity_sorts[et] = DeclareSort(et)
        self._eval_sorts = entity_sorts
        ns['entity_sorts'] = entity_sorts

        # Z3 builtins needed by eval'd strings
        ns.update({
            'Function': Function, 'BoolSort': BoolSort, 'IntSort': IntSort,
            'DeclareSort': DeclareSort, 'Const': Const,
            'ForAll': ForAll, 'Exists': Exists,
            'Implies': Implies, 'And': And, 'Or': Or, 'Not': Not,
        })

        # Create z3 variables ONCE
        z3_vars = {vn: Const(vn, entity_sorts["Entity"]) for vn in self.z3_vars}
        self._eval_z3_vars = z3_vars
        ns['z3_vars'] = z3_vars

        # Create fixed entities ONCE
        self._eval_entities = {}
        for et in entity_sorts:
            n = self.fov_entities[et]
            self._eval_entities[et] = [Const(f"{et}_{i}", entity_sorts[et]) for i in range(n)]

        # Create constants (e.g. Direction constants)
        constants_dict = self.data.get('Constants', None)
        if constants_dict:
            for const_name, const_type in constants_dict.items():
                if const_type in entity_sorts:
                    ns[const_name] = Const(const_name, entity_sorts[const_type])

        # Pre-evaluate predicate Z3 Function objects ONCE
        local_predicates = {}
        for pred_name, pred_info in self.predicates.items():
            eval_pred = eval(pred_info["instance"], ns)
            local_predicates[pred_name] = {
                "instance": eval_pred,
                "arity": pred_info["arity"],
                "function": pred_info["function"],
            }
        self._eval_predicates = local_predicates
        ns['local_predicates'] = local_predicates

        # Cache domain type names per predicate
        self._pred_domain_types = {}
        for pred_name, p in local_predicates.items():
            ep = p["instance"]
            self._pred_domain_types[pred_name] = [ep.domain(i).name() for i in range(p["arity"])]

        # Pre-evaluate rule Z3 expressions ONCE
        entity = self._eval_entities["Entity"][0]
        ns['entity'] = entity
        self._eval_rule_exprs = {}
        for rule_name, rule_template in self.rules['Task'].items():
            self._eval_rule_exprs[rule_name] = eval(rule_template["content"], ns)

        # Pre-build closed-world constraints ONCE
        self._eval_closed_world = []
        for var_name, z3_var in z3_vars.items():
            constraint = Or([z3_var == e for e in self._eval_entities['Entity']])
            self._eval_closed_world.append(ForAll([z3_var], constraint))

        # Bucket rule names by their immediate flag, so an eval() pass that has
        # no rules to check can return before touching the solver at all.
        self._eval_rules_by_immediate = {True: [], False: []}
        for rule_name, rule_template in self.rules['Task'].items():
            self._eval_rules_by_immediate[bool(rule_template["immediate"])].append(rule_name)

        # last_obs_dict keys read by _eval_fast, in the order it reads them.
        # Together with the action and the immediate flag they are its entire
        # input, which is what makes the result cache below sound.
        self._eval_grounding_keys = []
        for pred_name, p in local_predicates.items():
            fn = p["function"]
            if fn is None or fn == "None":
                continue
            k = 0
            if p["arity"] == 1:
                for _ in self._eval_entities[self._pred_domain_types[pred_name][0]]:
                    self._eval_grounding_keys.append("{}_{}".format(pred_name, k))
                    k += 1
            elif p["arity"] == 2:
                d0, d1 = self._pred_domain_types[pred_name]
                for _ in self._eval_entities[d0]:
                    for _ in self._eval_entities[d1]:
                        self._eval_grounding_keys.append("{}_{}".format(pred_name, k))
                        k += 1
        self._eval_result_cache = {}

        # Single reusable solver
        self._eval_solver = Solver()

        # Pre-warm grounding function cache
        for pred_info in self.predicates.values():
            fn = pred_info.get("function")
            if fn and fn != "None":
                _get_grounding_fn(fn)

        logger.info("Eval cache initialized successfully")

    def _eval_fast(self, rl_action, immediate):
        """Optimized eval using cached Z3 objects and single solver with push/pop."""
        last_obs_dict = self.last_rl_obs["last_obs_dict"]
        rule_tem = self.rules['Task']
        immediate = bool(immediate)

        # Nothing to check for this immediate flag: the rule loop below would not
        # execute, so the grounding asserts would be thrown away unused.
        if not self._eval_rules_by_immediate[immediate]:
            return False, 0, []

        # The outcome depends only on the grounded predicates, the action and the
        # immediate flag, all drawn from small finite sets -- so memoise it and
        # skip the Z3 round trip on the (many) repeated states.
        action_name = get_action_name(rl_action)
        cache_key = (immediate, action_name,
                     tuple(last_obs_dict[k] for k in self._eval_grounding_keys))
        cached = self._eval_result_cache.get(cache_key)
        if cached is not None:
            fail, reward, violations = cached
            return fail, reward, list(violations)

        solver = self._eval_solver
        solver.reset()

        # Add pre-built closed-world constraints
        for cw in self._eval_closed_world:
            solver.add(cw)

        # Add grounding constraints from last observation
        for pred_name, p in self._eval_predicates.items():
            eval_pred = p["instance"]
            fn = p["function"]

            if fn is None or fn == "None":
                # Action predicate - ground from rl_action
                if pred_name == action_name:
                    solver.add(eval_pred(self._eval_entities["Entity"][0]))
                else:
                    solver.add(Not(eval_pred(self._eval_entities["Entity"][0])))
                continue

            k = 0
            arity = p["arity"]
            if arity == 1:
                domain_name = self._pred_domain_types[pred_name][0]
                for entity in self._eval_entities[domain_name]:
                    if last_obs_dict[f"{pred_name}_{k}"]:
                        solver.add(eval_pred(entity))
                    else:
                        solver.add(Not(eval_pred(entity)))
                    k += 1
            elif arity == 2:
                d0 = self._pred_domain_types[pred_name][0]
                d1 = self._pred_domain_types[pred_name][1]
                for e1 in self._eval_entities[d0]:
                    for e2 in self._eval_entities[d1]:
                        if last_obs_dict[f"{pred_name}_{k}"]:
                            solver.add(eval_pred(e1, e2))
                        else:
                            solver.add(Not(eval_pred(e1, e2)))
                        k += 1

        # Check each rule using push/pop (1 solver instead of N)
        fail = False
        reward = 0
        violations = []
        for rule_name, rule_template in rule_tem.items():
            if rule_template["immediate"] == immediate:
                solver.push()
                solver.add(self._eval_rule_exprs[rule_name])
                if solver.check() != sat:
                    if rule_template["dead"]:
                        fail = True
                    reward += rule_template["reward"]
                    violations.append(rule_name)
                solver.pop()

        # Bounded: an unusually large entity/predicate count could otherwise make
        # the key space big enough to matter.
        if len(self._eval_result_cache) < self._EVAL_CACHE_MAX:
            self._eval_result_cache[cache_key] = (fail, reward, tuple(violations))
        return fail, reward, violations

    def _solve_grounding_fast(self, ego_name, partial_agents, partial_world, partial_intersections, partial_directions):
        """Fast grounding for RL agent - bypasses Z3 object creation entirely."""
        grounding = []
        grounding_dic = {}

        # Build entity name list matching world2entity ordering (ego first)
        entity_names_by_type = {"Entity": []}
        ego_entity_name = None
        for key, agent in partial_agents.items():
            if "ego" in key:
                ego_entity_name = f"Entity_{agent.type}_{agent.layer_id}"
                continue
            if "PH" in key:
                entity_names_by_type["Entity"].append(f"Entity_PH_{agent.layer_id}")
            else:
                entity_names_by_type["Entity"].append(f"Entity_{agent.type}_{agent.layer_id}")
        assert ego_entity_name is not None
        entity_names_by_type["Entity"] = [ego_entity_name] + entity_names_by_type["Entity"]

        for pred_name, pred_info in self.predicates.items():
            k = 0
            fn_name = pred_info["function"]
            if fn_name is None or fn_name == "None":
                continue

            method = _get_grounding_fn(fn_name)
            arity = pred_info["arity"]
            domain_types = self._pred_domain_types[pred_name]

            if arity == 1:
                for ename in entity_names_by_type.get(domain_types[0], []):
                    value = method(partial_world, partial_intersections, partial_directions, partial_agents, ename)
                    val = 1 if value else 0
                    grounding_dic[f"{pred_name}_{k}"] = val
                    grounding.append(val)
                    k += 1
            elif arity == 2:
                for e1 in entity_names_by_type.get(domain_types[0], []):
                    for e2 in entity_names_by_type.get(domain_types[1], []):
                        value = method(partial_world, partial_intersections, partial_directions, partial_agents, e1, e2)
                        val = 1 if value else 0
                        grounding_dic[f"{pred_name}_{k}"] = val
                        grounding.append(val)
                        k += 1

        result = {
            f"{ego_name}_grounding": np.array(grounding, dtype=np.float32),
            f"{ego_name}_grounding_dic": grounding_dic
        }
        assert len(grounding) == self.rl_input_shape
        return result

def logic_grounding_shape(
                      entity_types, 
                      predicates, 
                      var_names,
                      fov_entities):
    n = 0
    pred_grounding_index = {}
    # 1. create sorts and variables
    entity_sorts = {}
    for entity_type in entity_types:
        entity_sorts[entity_type] = DeclareSort(entity_type)
        assert fov_entities[entity_type] > 0, "Make sure the entity type (defined in rules) is in the fov_entities"
    # 3. entities
    entities = {}
    for entity_type in entity_sorts.keys():
        entity_num = fov_entities[entity_type]
        entities[entity_type] = [Const(f"{entity_type}_{i}", entity_sorts[entity_type]) for i in range(entity_num)]
    # 4. create, ground predicates and add to solver
    local_predicates = copy.deepcopy(predicates)
    for pred_name, pred_info in local_predicates.items():
        eval_pred = eval(pred_info["instance"])
        pred_info["instance"] = eval_pred
        arity = pred_info["arity"]

        method_full_name = pred_info["function"]
        if method_full_name == "None":
            continue

        if arity == 1:
            n_start = n
            # Unary predicate grounding
            for _ in entities[eval_pred.domain(0).name()]:
                n += 1
            pred_grounding_index[pred_name] = (n_start, n)
        elif arity == 2:
            n_start = n
            # Binary predicate grounding
            for _ in entities[eval_pred.domain(0).name()]:
                for _ in entities[eval_pred.domain(1).name()]:
                    n += 1
            pred_grounding_index[pred_name] = (n_start, n)
    logger.info("Given Predicates {}, the FOV entities {}, The logic grounding shape is: {}".format(local_predicates, fov_entities, n))
    return n, pred_grounding_index

def solve_sub_problem(ego_name, 
                      ego_action_mapping,
                      ego_action_dist,
                      rule_tem, 
                      entity_types, 
                      predicates, 
                      var_names,
                      partial_agents, 
                      partial_world, 
                      partial_intersections,
                      partial_directions,
                      fov_entities,
                      rl_flag,
                      constants,
                      rl_input_shape=None):
    grounding = []
    grounding_dic = {}
    # 1. create sorts and variables
    entity_sorts = {}
    for entity_type in entity_types:
        entity_sorts[entity_type] = DeclareSort(entity_type)
    z3_vars = {var_name: Const(var_name, entity_sorts["Entity"]) \
                       for var_name in var_names}
    # 2. partial world to entities
    local_entities = world2entity(entity_sorts, partial_intersections, partial_agents, fov_entities, rl_flag, constants)
    # 3. create, ground predicates and add to solver
    local_predicates = {name: {**info} for name, info in predicates.items()}
    # 4. create, ground predicates and add to solver
    if not rl_flag:
        local_solver = Solver()
        for pred_name, pred_info in local_predicates.items():
            eval_pred = eval(pred_info["instance"])
            pred_info["instance"] = eval_pred
            arity = pred_info["arity"]

            # Import the grounding method
            method_full_name = pred_info["function"]
            if method_full_name == "None":
                continue
            method = _get_grounding_fn(method_full_name)

            if arity == 1:
                # Unary predicate grounding
                for entity in local_entities[eval_pred.domain(0).name()]:
                    entity_name = entity.decl().name()
                    value = method(partial_world, partial_intersections, partial_directions, partial_agents, entity_name)
                    if value:
                        local_solver.add(eval_pred(entity))
                    else:
                        local_solver.add(Not(eval_pred(entity)))
            elif arity == 2:
                # Binary predicate grounding
                for entity1 in local_entities[eval_pred.domain(0).name()]:
                    entity1_name = entity1.decl().name()
                    for entity2 in local_entities[eval_pred.domain(1).name()]:
                        entity2_name = entity2.decl().name()
                        value = method(partial_world, partial_intersections, partial_directions, partial_agents, entity1_name, entity2_name)
                        if value:
                            local_solver.add(eval_pred(entity1, entity2))
                        else:
                            local_solver.add(Not(eval_pred(entity1, entity2)))
        # 5. create, ground rules and add to solver
        entity = local_entities["Entity"][0]
        for rule_name, rule_template in rule_tem.items():
            # Replace placeholder in the rule template with the actual agent entity
            instantiated_rule = eval(rule_template)
            local_solver.add(instantiated_rule)

        # **Important: Closed world quantifier rule, to ensure z3 do not add new entity to satisfy the rule and "dummy" is not part of the world**
        for var_name, z3_var in z3_vars.items():
            entity_list = local_entities["Entity"]
            constraint = Or([z3_var == entity for entity in entity_list])
            local_solver.add(ForAll([z3_var], constraint))
        
        # 6. solve
        if local_solver.check() == sat:
            model = local_solver.model()
            # Interpret the solution to the FOL problem
            action_mapping = ego_action_mapping
            action_dist = torch.zeros_like(ego_action_dist)

            for key in local_predicates.keys():
                action = []
                for action_id, action_name in action_mapping.items():
                    if key in action_name:
                        action.append(action_id)
                if len(action)>0:
                    for a in action:
                        if is_true(model.evaluate(local_predicates[key]["instance"](local_entities["Entity"][0]))):
                            action_dist[a] = 1.0
            # No action specified, use the default action, Normal
            if action_dist.sum() == 0:
                for action_id, action_name in action_mapping.items():
                    if "Normal" in action_name:
                        action_dist[action_id] = 1.0

            agents_actions = {ego_name: action_dist}
            return agents_actions
        else:
            action_mapping = ego_action_mapping
            action_dist = torch.zeros_like(ego_action_dist)

            for action_id, action_name in action_mapping.items():
                if "Normal" in action_name:
                    action_dist[action_id] = 1.0

            agents_actions = {ego_name: action_dist}
            return agents_actions
    else:
        for pred_name, pred_info in local_predicates.items():
            k = 0
            eval_pred = eval(pred_info["instance"])
            pred_info["instance"] = eval_pred
            arity = pred_info["arity"]

            # Import the grounding method
            method_full_name = pred_info["function"]

            if method_full_name == "None":
                continue

            method = _get_grounding_fn(method_full_name)

            if arity == 1:
                # Unary predicate grounding
                for entity in local_entities[eval_pred.domain(0).name()]:
                    entity_name = entity.decl().name()
                    value = method(partial_world, partial_intersections, partial_directions, partial_agents, entity_name)
                    if value:
                        grounding_dic["{}_{}".format(pred_name, k)] = 1
                        grounding.append(1)
                        k += 1
                    else:
                        grounding_dic["{}_{}".format(pred_name, k)] = 0
                        grounding.append(0)
                        k += 1
            elif arity == 2:
                # Binary predicate grounding
                for entity1 in local_entities[eval_pred.domain(0).name()]:
                    entity1_name = entity1.decl().name()
                    for entity2 in local_entities[eval_pred.domain(1).name()]:
                        entity2_name = entity2.decl().name()
                        value = method(partial_world, partial_intersections, partial_directions, partial_agents, entity1_name, entity2_name)
                        if value:
                            grounding_dic["{}_{}".format(pred_name, k)] = 1
                            grounding.append(1)
                            k += 1
                        else:
                            grounding_dic["{}_{}".format(pred_name, k)] = 0
                            grounding.append(0)
                            k += 1

        agents_actions = {
            "{}_grounding".format(ego_name): np.array(grounding, dtype=np.float32),
            "{}_grounding_dic".format(ego_name): grounding_dic
        }
        assert len(grounding) == rl_input_shape

        return agents_actions

def eval_action(rl_action,
                rule_tem, 
                entity_types, 
                predicates, 
                var_names,
                fov_entities,
                last_obs_dict,
                last_obs,
                constants,
                immediate=False):
    grounding = []
    # 1. create sorts and variables
    entity_sorts = {}
    
    for entity_type in entity_types:
        entity_sorts[entity_type] = DeclareSort(entity_type)
    z3_vars = {var_name: Const(var_name, entity_sorts["Entity"]) \
                       for var_name in var_names}
    
    if constants is not None:
        for const_name, const_type in constants.items():
            if const_type == "Direction":
                exec(f"{const_name} = Const('{const_name}', entity_sorts['{const_type}'])")
    # 2. entities
    entities = {}
    for entity_type in entity_sorts.keys():
        entity_num = fov_entities[entity_type]
        entities[entity_type] = [Const(f"{entity_type}_{i}", entity_sorts[entity_type]) for i in range(entity_num)]
    # 3. create, ground predicates and add to solver
    local_predicates = copy.deepcopy(predicates)
    # 4. create, ground predicates and add to solver
    local_solvers = {rule_name: Solver() for rule_name in rule_tem.keys()}
    for pred_name, pred_info in local_predicates.items():
        k = 0
        eval_pred = eval(pred_info["instance"])
        pred_info["instance"] = eval_pred
        arity = pred_info["arity"]

        # Import the grounding method
        method_full_name = pred_info["function"]

        if method_full_name == "None":
            assert rl_action is not None, "Make sure the rl_action is not None"
            # ego action
            action_name = get_action_name(rl_action)
            if pred_name == action_name:
                for rule_name, rule_template in rule_tem.items():
                    local_solvers[rule_name].add(eval_pred(entities["Entity"][0]))
            else:
                for rule_name, rule_template in rule_tem.items():
                    local_solvers[rule_name].add(Not(eval_pred(entities["Entity"][0])))
            continue

        if arity == 1:
            # Unary predicate grounding
            for entity in entities[eval_pred.domain(0).name()]:
                if last_obs_dict["{}_{}".format(pred_name, k)]:
                    grounding.append(1)
                    k += 1
                    for rule_name, rule_template in rule_tem.items():
                        local_solvers[rule_name].add(eval_pred(entity))
                else:
                    grounding.append(0)
                    k += 1
                    for rule_name, rule_template in rule_tem.items():
                        local_solvers[rule_name].add(Not(eval_pred(entity)))
        elif arity == 2:
            # Binary predicate grounding
            for entity1 in entities[eval_pred.domain(0).name()]:
                for entity2 in entities[eval_pred.domain(1).name()]:
                    if last_obs_dict["{}_{}".format(pred_name, k)]:
                        grounding.append(1)
                        k += 1
                        for rule_name, rule_template in rule_tem.items():
                            local_solvers[rule_name].add(eval_pred(entity1, entity2))
                    else:
                        grounding.append(0)
                        k += 1
                        for rule_name, rule_template in rule_tem.items():
                            local_solvers[rule_name].add(Not(eval_pred(entity1, entity2)))

    # 5. create, ground rules and add to solver
    local_rule_tem = copy.deepcopy(rule_tem)
    for rule_name, rule_template in local_rule_tem.items():
        # the first entity is the ego agent
        entity = entities["Entity"][0]
        # Replace placeholder in the rule template with the actual agent entity
        instantiated_rule = eval(rule_template["content"])
        local_solvers[rule_name].add(instantiated_rule)

    # **Important: Closed world quantifier rule, to ensure z3 do not add new entity to satisfy the rule and "dummy" is not part of the world**
    for var_name, z3_var in z3_vars.items():
        entity_list = entities['Entity']
        for rule_name, rule_template in rule_tem.items():
            constraint = Or([z3_var == entity for entity in entity_list])
            local_solvers[rule_name].add(ForAll([z3_var], constraint))
    
    # 6. solve for reward
    obs = np.array(grounding, dtype=np.float32)
    assert np.all(obs == last_obs), (obs, last_obs)
    fail = False
    reward = 0
    violations = []
    for rule_name, rule_solver in local_solvers.items():
        if rule_tem[rule_name]["immediate"] == immediate:
            if rule_solver.check() == sat:
                    continue
            else:
                if rule_tem[rule_name]["dead"]:
                    fail = True
                reward += local_rule_tem[rule_name]["reward"]
                violations.append(rule_name)

    return fail, reward, violations

def get_action_name(rl_action):
    # see agents/car.py
    if rl_action[0] == 1:
        return "Slow"
    elif rl_action[4] == 1:
        return "Normal"
    elif rl_action[8] == 1:
        return "Fast"
    else:
        return "Stop"

def split_into_batches(keys, batch_size):
    """Split keys into batches of a given size."""
    for i in range(0, len(keys), batch_size):
        yield keys[i:i + batch_size]

def world2entity(entity_sorts, partial_intersect, partial_agents, fov_entities, rl_flag, constants):
    # all the enitities are stored in self.entities
    entities = {}
    for entity_type in entity_sorts.keys():
        entities[entity_type] = []
        flag = False
        # For Agents
        if entity_type == "Entity":
            for key, agent in partial_agents.items():
                if "ego" in key:
                    ego_agent = agent
                    flag = True
                    continue
                if "PH" in key:
                    agent_id = agent.layer_id
                    agent_name = f"Entity_PH_{agent_id}"
                else:
                    agent_id = agent.layer_id
                    agent_type = agent.type
                    agent_name = f"Entity_{agent_type}_{agent_id}"
                # Create a Z3 constant for the agent
                agent_entity = Const(agent_name, entity_sorts['Entity'])
                entities[entity_type].append(agent_entity)
            assert flag, logger.info(partial_agents)
            agent_id = ego_agent.layer_id
            agent_type = ego_agent.type
            agent_name = f"Entity_{agent_type}_{agent_id}"
            # Create a Z3 constant for the agent
            agent_entity = Const(agent_name, entity_sorts['Entity'])
            # ego agent is the first
            entities[entity_type] = [agent_entity] + entities[entity_type]
            if rl_flag:
                assert len(entities[entity_type]) == fov_entities["Entity"], logger.info(entities)
    return entities
