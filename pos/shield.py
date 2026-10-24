"""Probabilistic Logic Shield (PLS, Yang et al. 2023).

A ProbLog program maps sensor readings and a candidate action to
``P(safe | s, a)`` through the query ``safe_next``. The program is compiled once
into an arithmetic circuit (via kompyle) and evaluated by weighted model
counting. The program must declare its inputs as annotated disjunctions over
``sensor_value(i)`` and ``action(i)`` (see ``configs/*/env/shield.pl``).
"""
import functools
from typing import Callable, Optional

import kompyle
import torch as th
from problog.formula import LogicDAG, LogicFormula, Term
from problog.logic import Constant


class SafetyCircuit:
    """Compiled ``safe_next`` query of a ProbLog shield program.

    Literals ``1..n_sensors`` are the sensor values, the next ``n_actions``
    literals the action probabilities; every other probabilistic fact of the
    program keeps its constant weight.
    """

    def __init__(self, program: str, n_actions: int, n_sensors: int):
        dag = LogicDAG.create_from(LogicFormula.create_from(program + "\n\nquery(safe_next)."))
        sensor_lits = [kompyle.lit(i + 1) for i in range(n_sensors)]
        action_lits = [kompyle.lit(n_sensors + i + 1) for i in range(n_actions)]

        refs = [None] * len(dag)

        def ref(i):
            r = refs[abs(i) - 1]
            return ~r if i < 0 else r

        n_lits = n_sensors + n_actions
        weights = []
        weight_index = {}
        for index, node, ntype in dag:
            if ntype == "atom":
                prob = node.probability
                if isinstance(prob, Term) and prob.functor == "sensor_value":
                    refs[index - 1] = sensor_lits[prob.args[0].value]
                elif isinstance(prob, Term) and prob.functor == "action":
                    refs[index - 1] = action_lits[prob.args[0].value]
                else:
                    n_lits += 1
                    refs[index - 1] = kompyle.lit(n_lits)
                    weights.append(self._to_weight(prob))
                    weight_index[index] = len(weights) - 1
            elif ntype == "disj":
                refs[index - 1] = functools.reduce(lambda x, y: x | y, (ref(c) for c in node.children))
            elif ntype == "conj":
                refs[index - 1] = functools.reduce(lambda x, y: x & y, (ref(c) for c in node.children))
            else:
                raise ValueError(f"Unsupported node type {ntype!r} in the shield program.")

        # Annotated disjunctions become hard constraints conjoined to the formula.
        constraints = kompyle.one()
        for constraint in dag.constraints():
            clauses = constraint.as_clauses()
            if clauses:
                weights[weight_index[constraint.extra_node]] = 0.0
                clauses = [functools.reduce(lambda x, y: x | y, [ref(lit) for lit in c]) for c in clauses]
                constraints &= functools.reduce(lambda x, y: x & y, clauses)
        refs[-1] = refs[-1] & constraints

        formula = ref(dag.get_node_by_name(Term("safe_next")))
        self._circuit = formula.compile(formula.nb_vars())
        self._weights = th.tensor(weights)

    @staticmethod
    def _to_weight(term):
        if isinstance(term, Constant):
            return term.value
        if isinstance(term, bool):
            return 1.0 if term else 0.0
        raise ValueError("Atom labelled with an unknown probability type.")

    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        """``inputs``: (batch, n_sensors + n_actions) -> (batch,) probabilities."""
        weights = th.cat([inputs, self._weights.to(inputs).expand(inputs.size(0), -1)], dim=-1)
        rows = weights.detach().cpu().numpy()
        probs = [self._circuit._wmc_grad(row.tolist())[0] for row in rows]
        return th.tensor(probs, device=inputs.device, dtype=inputs.dtype)


class Shield:
    """Action-level PLS shield.

    :param prolog_path: ProbLog program defining ``safe_next``.
    :param get_sensor_values: callable returning the current (1, n_sensors)
        sensor tensor of the environment.
    """

    def __init__(self, prolog_path: str, get_sensor_values: Optional[Callable] = None, device="cpu"):
        with open(prolog_path) as f:
            program = f.read()
        lines = program.split("\n")
        self.num_sensors = len([line for line in lines if "sensor_value" in line])
        self.num_actions = len([line for line in lines if line.startswith("action(")])
        self.get_sensor_values = get_sensor_values
        self.device = th.device(device)
        self.circuit = SafetyCircuit(program, self.num_actions, self.num_sensors)

    def action_safeties(self, sensor_values: th.Tensor) -> th.Tensor:
        """``P(safe | s, a)`` for every action: (batch, n_sensors) -> (batch, n_actions)."""
        sensor_values = sensor_values.to(self.device)
        batch = sensor_values.size(0)
        # One circuit call for all (action, state) pairs, laid out action-major.
        sensors = sensor_values.repeat(self.num_actions, 1)
        actions = th.eye(self.num_actions, device=self.device).repeat_interleave(batch, dim=0)
        flat = self.circuit(th.cat([sensors, actions], dim=-1))
        return flat.reshape(self.num_actions, batch).t().contiguous()

    def shield_policy(self, base_probs: th.Tensor, sensor_values: th.Tensor) -> th.Tensor:
        """PLS shielded policy ``pi+(a|s) = P(safe|s,a) pi(a|s) / P_pi(safe|s)``."""
        base_probs = base_probs.to(self.device)
        weighted = self.action_safeties(sensor_values) * base_probs
        return weighted / th.sum(weighted, dim=-1).unsqueeze(-1)
