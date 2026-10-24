import torch
import torch.nn.functional as F
from ..core.config import *

def gen_occ(agent_state_matrix):
    occ_map = torch.zeros_like(agent_state_matrix[0])
    car_kernel = torch.ones((CAR_SIZE, CAR_SIZE))

    # Create masks for each agent type
    pedestrian_mask = torch.zeros_like(agent_state_matrix, dtype=torch.bool)
    car_mask = torch.zeros_like(agent_state_matrix, dtype=torch.bool)
    agent_dict = {
        'Pedestrian': TYPE_MAP['Pedestrian'],
        'Car': TYPE_MAP['Car']
    }
    for agent_type, label in agent_dict.items():
        agent_mask = agent_state_matrix == label
        if agent_type == 'Pedestrian':
            pedestrian_mask |= agent_mask
        elif agent_type == 'Car':
            car_mask |= agent_mask

    # Aggregate pedestrian occupancy
    pedestrian_occ = pedestrian_mask.any(dim=0)
    if pedestrian_occ.any():
        occ_map += pedestrian_occ.to(occ_map.dtype)

    # Aggregate car occupancy and apply convolution
    car_occ = car_mask.any(dim=0).to(torch.float32).unsqueeze(0).unsqueeze(0)
    if car_occ.any():
        enlarged_car_occ = F.conv2d(car_occ, car_kernel.unsqueeze(0).unsqueeze(0), padding=CAR_SIZE//2)
        enlarged_car_occ = enlarged_car_occ.squeeze(0).squeeze(0) > 0
        occ_map += enlarged_car_occ.to(occ_map.dtype)
    return occ_map
