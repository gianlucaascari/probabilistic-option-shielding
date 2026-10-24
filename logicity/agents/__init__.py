from .basic import Agent
from .pedestrian import Pedestrian
from .hierarchical_car import HierarchicalCar

Agent_mapper = {
    'Pedestrian': Pedestrian,
    'Hierarchical_car': HierarchicalCar,
}
