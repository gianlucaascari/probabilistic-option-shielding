import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ...core.config import *
import logging

logger = logging.getLogger(__name__)

TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}

def IsCar(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    if "PH" in entity:
        return 0
    if "Car" in entity:
        return 1
    else:
        return 0
    
def IsPed(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    if "PH" in entity:
        return 0
    if "Pedestrian" in entity:
        return 1
    else:
        return 0

def IsAmb(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "ambulance" in agent_concept:
        return 1
    else:
        return 0

def IsBus(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "bus" in agent_concept:
        return 1
    else:
        return 0
    
def IsTiro(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    assert "Agent" in entity
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "tiro" in agent_concept:
        return 1
    else:
        return 0
    
def IsPolice(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "police" in agent_concept:
        return 1
    else:
        return 0

def IsTiro(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "tiro" in agent_concept:
        return 1
    else:
        return 0

def IsReckless(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "reckless" in agent_concept:
        return 1
    else:
        return 0
    
def IsOld(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    if "Car" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "old" in agent_concept:
        return 1
    else:
        return 0
    
def IsYoung(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    if "Car" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "young" in agent_concept:
        return 1
    else:
        return 0
    
def IsAtInter(world_matrix, intersect_matrix, direction_matrix, agents, entity1):
    if "PH" in entity1:
        return 0

    _, agent_type, layer_id = entity1.split("_")
    layer_id = int(layer_id)
    agent_layer = world_matrix[layer_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    # at intersection needs to care if the car is "entering" or "leaving", so use intersect_matrix[0]
    if agent_type == "Car":
        if intersect_matrix[0, agent_position[0], agent_position[1]]:
            return 1
        else:
            return 0
    else:
        if intersect_matrix[1, agent_position[0], agent_position[1]]:
            return 1
        else:
            return 0

def IsInInter(world_matrix, intersect_matrix, direction_matrix, agents, entity1):
    if "PH" in entity1:
        return 0
    _, agent_type, layer_id = entity1.split("_")
    layer_id = int(layer_id)
    agent_layer = world_matrix[layer_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    if intersect_matrix[2, agent_position[0], agent_position[1]]:
        return 1
    else:
        return 0

def IsClose(world_matrix, intersect_matrix, direction_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    layer_id1 = int(layer_id1)
    layer_id2 = int(layer_id2)
    agent_layer1 = world_matrix[layer_id1]
    agent_layer2 = world_matrix[layer_id2]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    eudis = torch.sqrt(torch.sum((agent_position1 - agent_position2)**2))
    if eudis > CLOSE_RANGE_MIN and eudis <= CLOSE_RANGE_MAX:
        return 1
    else:
        return 0

def HigherPri(world_matrix, intersect_matrix, direction_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    
    _, _, agent_layer1 = entity1.split("_")
    _, _, agent_layer2 = entity2.split("_")

    if agent_layer1 in agents.keys():
        agent_prio1 = agents[agent_layer1].priority
    else:
        assert "ego_{}".format(agent_layer1) in agents.keys()
        agent_prio1 = agents["ego_{}".format(agent_layer1)].priority

    if agent_layer2 in agents.keys():
        agent_prio2 = agents[agent_layer2].priority
    else:
        assert "ego_{}".format(agent_layer2) in agents.keys()
        agent_prio2 = agents["ego_{}".format(agent_layer2)].priority

    if agent_prio1 < agent_prio2:
        return 1
    else:
        return 0

def CollidingClose(world_matrix, intersect_matrix, direction_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    # 1. Get the position of the two agents
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    agent_layer1 = world_matrix[int(layer_id1)]
    agent_layer2 = world_matrix[int(layer_id2)]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    # 2. Get the moving direction of the first agent
    if layer_id1 in agents.keys():
        agent1_dire = agents[layer_id1].moving_direction
    else:
        assert "ego_{}".format(layer_id1) in agents.keys()
        agent1_dire = agents["ego_{}".format(layer_id1)].moving_direction
    if agent1_dire == None:
        return 0
    else:
        dist = torch.sqrt(torch.sum((agent_position1 - agent_position2)**2))
        if dist > OCC_CHECK_RANGE[agent_type1]:
            return 0
        elif dist == 0:
            return np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            agent1_dire_vec = torch.tensor(DIRECTION_VECTOR[agent1_dire])
            angle = torch.acos(torch.dot(agent1_dire_vec, (agent_position2 - agent_position1)) / dist)
            if angle < OCC_CHECK_ANGEL:
                if agent_type1 == "Car":
                    # Cars will definitely stop to avoid collide
                    return 1
                else:
                    # Pedestrians will probably stop to avoid collide
                    sample = np.random.rand()
                    if sample < PED_AGGR:
                        return 1
                    else:
                        return 0
    return 0

def LeftOf(world_matrix, intersect_matrix, direction_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    # 1. Get the position of the two agents
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    agent_layer1 = world_matrix[int(layer_id1)]
    agent_layer2 = world_matrix[int(layer_id2)]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    # 2. note: entity1 is on the left of entity2, so we get the direction of entity2
    if layer_id2 in agents.keys():
        agent2_dire = agents[layer_id2].moving_direction
    else:
        assert "ego_{}".format(layer_id2) in agents.keys()
        agent2_dire = agents["ego_{}".format(layer_id2)].moving_direction
    if agent2_dire == None:
        return 0
    else:
        agent2_dire_vec = torch.tensor(DIRECTION_VECTOR[agent2_dire])
        relative_pos = agent_position1 - agent_position2
        dx, dy = agent2_dire_vec # Direction Agent 2 is facing
        rx, ry = relative_pos # Vector from Agent 2 to Agent 1
        z_component = dx * ry - dy * rx
        if z_component > 0:
            return 1
        else:
            return 0

def RightOf(world_matrix, intersect_matrix, direction_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    # 1. Get the position of the two agents
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    agent_layer1 = world_matrix[int(layer_id1)]
    agent_layer2 = world_matrix[int(layer_id2)]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    # 2. note: entity1 is on the right of entity2, so we get the direction of entity2
    if layer_id2 in agents.keys():
        agent2_dire = agents[layer_id2].moving_direction
    else:
        assert "ego_{}".format(layer_id2) in agents.keys()
        agent2_dire = agents["ego_{}".format(layer_id2)].moving_direction
    if agent2_dire == None:
        return 0
    else:
        agent2_dire_vec = torch.tensor(DIRECTION_VECTOR[agent2_dire])
        relative_pos = agent_position1 - agent_position2
        dx, dy = agent2_dire_vec # Direction Agent 2 is facing
        rx, ry = relative_pos # Vector from Agent 2 to Agent 1
        z_component = dx * ry - dy * rx
        if z_component < 0:
            return 1
        else:
            return 0

def NextTo(world_matrix, intersect_matrix, direction_matrix, agents, entity1, entity2):
    # Next to checker, closer than Close checker
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    layer_id1 = int(layer_id1)
    layer_id2 = int(layer_id2)
    agent_layer1 = world_matrix[layer_id1]
    agent_layer2 = world_matrix[layer_id2]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    eudis = torch.sqrt(torch.sum((agent_position1 - agent_position2)**2))
    if eudis < CLOSE_RANGE_MIN:
        return 1
    else:
        return 0
    
    
OPPOSITE_DIRECTION_MATRIX = {
    #         u  r  d  l
    "Up":    [0, 1, 1, 1],
    "Right": [1, 0, 1, 1],
    "Down":  [1, 1, 0, 1],
    "Left":  [1, 1, 1, 0],
}

def AgainstTraffic(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    if "PH" in entity:
        return 0
    _, agent_type, layer_id = entity.split("_")
    
    agent_layer = world_matrix[int(layer_id)]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    allowed_directions = direction_matrix[:, agent_position[0], agent_position[1]]
    
    if layer_id in agents.keys():
        direction = agents[layer_id].moving_direction
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        direction = agents["ego_{}".format(layer_id)].moving_direction
        
    if direction == None:
        return 0
    
    # assume going in not allowed direction, if some other is allowed change idea
    # this is because in crossings you have 2 allowed directions, so you could be
    # opposite to one but sideways to the other
    result = 1
    for i, v in enumerate(allowed_directions):
        if v and not OPPOSITE_DIRECTION_MATRIX[direction][i]:
            result = 0
            break
    
    return result

    
SIDEWAYS_DIRECTION_MATRIX = {
    #         u  r  d  l
    "Up":    [0, 1, 0, 1],
    "Right": [1, 0, 1, 0],
    "Down":  [0, 1, 0, 1],
    "Left":  [1, 0, 1, 0],
}

def SidewaysToTraffic(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    if "PH" in entity:
        return 0
    _, agent_type, layer_id = entity.split("_")
    
    agent_layer = world_matrix[int(layer_id)]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    allowed_directions = direction_matrix[:, agent_position[0], agent_position[1]]
    
    if layer_id in agents.keys():
        direction = agents[layer_id].moving_direction
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        direction = agents["ego_{}".format(layer_id)].moving_direction
        
    if direction == None:
        return 0
    
    # assume going sideways, if you are right or opposide direction to some street
    # change your mind, this is because in crossings you have 2 allowed directions, 
    # so you could be sideways to one but right sided to the other
    result = 1
    for i, v in enumerate(allowed_directions):
        if v and not SIDEWAYS_DIRECTION_MATRIX[direction][i]:
            result = 0
            break
    
    return result

def Collided(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    if "PH" in entity:
        return 0
    _, agent_type, layer_id = entity.split("_")

    agent_layer = world_matrix[int(layer_id)]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]

    for j in range(3, world_matrix.shape[0]):
        if j != int(layer_id):
            val = world_matrix[j][agent_position[0]][agent_position[1]]
            if val != 0 and (val % 1.0) == 0.0:
                return 1
    return 0


# ---------------------------------------------------------------------------
# Military zone predicate
# ---------------------------------------------------------------------------

_MILITARY_MATRIX = None  # None → backward compat: InMilitaryZone always returns 0


def set_military_matrix(matrix):
    """Register a per-map military zone matrix (numpy array, dtype uint8)."""
    global _MILITARY_MATRIX
    _MILITARY_MATRIX = matrix


def InMilitaryZone(world_matrix, intersect_matrix, direction_matrix, agents, entity):
    """Return 1 if the entity's current cell is inside a military zone."""
    if "PH" in entity or _MILITARY_MATRIX is None:
        return 0
    _, agent_type, layer_id = entity.split("_")

    # Use the agent's stored global position to avoid indexing _MILITARY_MATRIX
    # with local FOV-crop coordinates (world_matrix is a spatial crop, not the full grid).
    if layer_id in agents:
        agent = agents[layer_id]
    else:
        agent = agents["ego_{}".format(layer_id)]
    r, c = int(agent.pos[0]), int(agent.pos[1])

    if r < _MILITARY_MATRIX.shape[0] and c < _MILITARY_MATRIX.shape[1]:
        return int(_MILITARY_MATRIX[r, c])
    return 0