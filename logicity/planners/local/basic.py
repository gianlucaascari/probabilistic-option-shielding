from yaml import load, FullLoader

class LocalPlanner:
    def __init__(self, logic_engine_file):        
        # Load the yaml file, create the predicates and rules
        onotology_yaml, rule_yaml = logic_engine_file['ontology'], logic_engine_file['rule']
        with open(onotology_yaml, 'r') as file:
            self.data = load(file, Loader=FullLoader)
        with open(rule_yaml, 'r') as file:
            self.data.update(load(file, Loader=FullLoader))
        
        self._create_entities()
        self._create_predicates()
        self._create_rules()
    
    def _create_entities(self):
        # Create the entities
        raise NotImplementedError
    
    def _create_predicates(self):
        # Create the predicates
        raise NotImplementedError
    
    def _create_rules(self):
        # Create the rules
        raise NotImplementedError
    
    def plan(self, map_data, start, goal):
        # Plan a path from start to goal
        pass

