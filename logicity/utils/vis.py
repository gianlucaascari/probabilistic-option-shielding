import numpy as np
import random
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import numpy as np
from skimage.transform import resize
from matplotlib.patches import Patch
from ..core.config import *

def visualize_city(city, resolution, agent_layer=None, file_name="city.png"):
    # Define a color for each entity
    # only visualize a static map with buildings, streets, and one layer agents
    # Create a visual grid of the city, buildings and streets
    visual_grid = np.ones((resolution, resolution, 3), dtype=np.uint8)*200
    np_grid = city.city_grid.numpy().astype(np.float32)
    scale_factor = resolution/city.grid_size[0]
    for k in range(1, 3):
        for i in range(city.grid_size[0]):
            for j in range(city.grid_size[1]):
                if np_grid[k][i][j] != 0:
                    color = city.color_map[np_grid[k][i][j]]
                    visual_grid[int(i*scale_factor):int((i+1)*scale_factor), int(j*scale_factor):int((j+1)*scale_factor)] = color

    # draw agent's if provided layer
    if agent_layer != None:
        visual_grid = vis_agent(visual_grid, city, agent_layer, scale_factor)
    
    # Add the legend
    padding = int(5*scale_factor)
    column_width = int(70*scale_factor)
    legend_width = resolution
    legend_height = resolution
    legend_item_height = int(10*scale_factor)
    legend_img = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255  # white background

    for idx, (key, value) in enumerate(city.color_map.items()):
        c = idx//23
        idx = idx%23
        y_offset = idx * legend_item_height + padding
        if y_offset + legend_item_height > legend_height:  # Ensure we don't render beyond the legend image
            break
        cv2.rectangle(legend_img, (padding+column_width*c, y_offset), (padding+column_width*c + legend_item_height, y_offset + legend_item_height), city.color_map[key], -1)
        if key in city.label2type.keys():
            cv2.putText(legend_img, str(city.label2type[key]), (padding+column_width*c + int(15*scale_factor), y_offset + legend_item_height - int(3*scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 0.2*scale_factor, \
                (0, 0, 0), 2, lineType=cv2.LINE_AA)
        else:
            cv2.putText(legend_img, key, (padding+column_width*c + int(15*scale_factor), y_offset + legend_item_height - int(3*scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 0.2*scale_factor, \
                (0, 0, 0), 2, lineType=cv2.LINE_AA)

    # Combine the visual grid and the legend side by side
    combined_img = np.hstack((visual_grid, legend_img))
    # just visualize the left corner that is constructed
    # combined_img = combined_img[:int(resolution/2), :int(resolution/2), :]

    # Use OpenCV to display the city
    cv2.imwrite(file_name, combined_img)

def vis_agent(vis_grid, city, agent_layer_ids, scale_factor, curr=True, s=True, g=True, path=True):
    if agent_layer_ids == -1:
        agent_layer_ids = list(range(BASIC_LAYER, city.city_grid.shape[0]))
    if isinstance(agent_layer_ids, list):
        # visualize multi-ple agents
        for agent_layer_id in agent_layer_ids:
            agent_type = city.agents[agent_layer_id-BASIC_LAYER].type
            agent_id = city.agents[agent_layer_id-BASIC_LAYER].id
            agent_layer = city.city_grid[agent_layer_id]
            agent_color = city.color_map["{}_{}".format(agent_type, agent_id)]
            # get points
            cur_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]).float()).tolist()
            planned_traj = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_GLOBAL_PATH_PLUS).float()).tolist()
            walked_traj = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_WALKED_PATH_PLUS).float()).tolist()
            goal_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_GOAL_PLUS).float()).tolist()
            start_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_START_PLUS).float()).tolist()
            # Path
            if len(planned_traj) > 0 and path:  
                for way_point in planned_traj:
                    vis_grid[int(way_point[0]*scale_factor):int((way_point[0]+1)*scale_factor), \
                        int(way_point[1]*scale_factor):int((way_point[1]+1)*scale_factor)] = agent_color
            if len(walked_traj) > 0 and path:  
                for way_point_ in walked_traj:
                    vis_grid[int(way_point_[0]*scale_factor):int((way_point_[0]+1)*scale_factor), \
                        int(way_point_[1]*scale_factor):int((way_point_[1]+1)*scale_factor)] = agent_color
            # Points
            if len(start_agent_pos) > 0 and s:
                cv2.drawMarker(vis_grid, (int((start_agent_pos[0][1]+0.5)*scale_factor), int((start_agent_pos[0][0]+0.5)*scale_factor)), \
                    agent_color, markerType=cv2.MARKER_TRIANGLE_UP, markerSize=int(scale_factor*0.8), thickness=3, line_type=cv2.LINE_AA)
            if len(goal_agent_pos) > 0 and g:   
                cv2.drawMarker(vis_grid, (int((goal_agent_pos[0][1]+0.5)*scale_factor), int((goal_agent_pos[0][0]+0.5)*scale_factor)), \
                    agent_color, markerType=cv2.MARKER_TRIANGLE_DOWN, markerSize=int(scale_factor*0.8), thickness=3, line_type=cv2.LINE_AA)
            assert len(cur_agent_pos) == 1
            if curr:
                cv2.drawMarker(vis_grid, (int((cur_agent_pos[0][1]+0.5)*scale_factor), int((cur_agent_pos[0][0]+0.5)*scale_factor)), \
                    agent_color, markerType=cv2.MARKER_DIAMOND, markerSize=int(scale_factor*0.8), thickness=3, line_type=cv2.LINE_AA)

    elif isinstance(agent_layer_ids, int):
        agent_layer_id = agent_layer_ids
        assert agent_layer_id >= BASIC_LAYER
        agent_type = city.agents[agent_layer_id-BASIC_LAYER].type
        agent_id = city.agents[agent_layer_id-BASIC_LAYER].id
        agent_layer = city.city_grid[agent_layer_id]
        agent_color = city.color_map["{}_{}".format(agent_type, agent_id)]
        # draw points
        cur_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]).float()).tolist()
        planned_traj = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_GLOBAL_PATH_PLUS).float()).tolist()
        walked_traj = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_WALKED_PATH_PLUS).float()).tolist()
        goal_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_GOAL_PLUS).float()).tolist()
        start_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]+AGENT_START_PLUS).float()).tolist()
        # Path
        if len(planned_traj) > 0 and path:  
            if len(planned_traj) > 0 and path:  
                for way_point in planned_traj:
                    vis_grid[int(way_point[0]*scale_factor):int((way_point[0]+1)*scale_factor), \
                        int(way_point[1]*scale_factor):int((way_point[1]+1)*scale_factor)] = agent_color
            if len(walked_traj) > 0 and path:  
                for way_point_ in walked_traj:
                    vis_grid[int(way_point_[0]*scale_factor):int((way_point[0]+1)*scale_factor), \
                        int(way_point_[1]*scale_factor):int((way_point[1]+1)*scale_factor)] = agent_color
            # Points
            if len(start_agent_pos) > 0 and s:
                cv2.drawMarker(vis_grid, (int((start_agent_pos[0][1]+0.5)*scale_factor), int((start_agent_pos[0][0]+0.5)*scale_factor)), \
                    agent_color, markerType=cv2.MARKER_TRIANGLE_UP, markerSize=int(scale_factor*0.8), thickness=3, line_type=cv2.LINE_AA)
            if len(goal_agent_pos) > 0 and g:   
                cv2.drawMarker(vis_grid, (int((goal_agent_pos[0][1]+0.5)*scale_factor), int((goal_agent_pos[0][0]+0.5)*scale_factor)), \
                    agent_color, markerType=cv2.MARKER_TRIANGLE_DOWN, markerSize=int(scale_factor*0.8), thickness=3, line_type=cv2.LINE_AA)
            assert len(cur_agent_pos) == 1
            if curr:
                cv2.drawMarker(vis_grid, (int((cur_agent_pos[0][1]+0.5)*scale_factor), int((cur_agent_pos[0][0]+0.5)*scale_factor)), \
                    agent_color, markerType=cv2.MARKER_DIAMOND, markerSize=int(scale_factor*0.8), thickness=3, line_type=cv2.LINE_AA)

    return vis_grid

def visualize_intersections(intersection_matrix, file_name="intersections.png"):
    # Get unique intersection IDs (excluding 0)
    unique_intersections = np.unique(intersection_matrix)
    unique_intersections = unique_intersections[unique_intersections != 0]
    
    # Create a color map for each intersection ID
    colors = list(mcolors.CSS4_COLORS.values())
    intersection_colors = {uid: colors[i % len(colors)] for i, uid in enumerate(unique_intersections)}

    # Create an RGB visualization matrix
    vis_matrix = np.zeros((*intersection_matrix.shape, 3), dtype=np.uint8)

    for uid, color in intersection_colors.items():
        r, g, b = mcolors.hex2color(color)
        mask = (intersection_matrix == uid)
        vis_matrix[mask] = (np.array([r, g, b]) * 255).astype(np.uint8)

    # Plot
    plt.imshow(vis_matrix)
    plt.title("Intersections Visualization")
    plt.axis('off')
    plt.imsave(file_name, vis_matrix)
    
def visualize_grid(grid, cmap='viridis', show_values=False, title="2D Grid"):
    """
    Visualize a 2D grid of numerical values using Matplotlib.

    Parameters:
        grid (2D list or np.ndarray): The grid to visualize.
        cmap (str): Colormap name (e.g. 'viridis', 'plasma', 'coolwarm').
        show_values (bool): Whether to display the numeric values on the grid.
        title (str): Title of the plot.
    """
    grid = np.array(grid)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=cmap, origin='upper')
    plt.colorbar(label='Value')

    if show_values:
        # Overlay the numeric values
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                plt.text(
                    j, i, f"{grid[i, j]}",
                    ha='center', va='center',
                    color='white' if grid[i, j] > np.max(grid)/2 else 'black'
                )

    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.show()
    
def visualize_boolean_grids(grids, colors=None, alpha=0.4, title="Overlayed Boolean Grids"):
    """
    Visualize multiple boolean 2D grids overlaid with transparency.

    Parameters:
        grids (list of 2D np.ndarray or lists): Each grid is a boolean matrix of the same shape.
        colors (list of str): List of color names or codes for each grid (default uses distinct colors).
        alpha (float): Transparency level for each layer (0.0 = fully transparent, 1.0 = opaque).
        title (str): Title for the plot.
    """
    n = len(grids)
    if n == 0:
        raise ValueError("No grids provided.")
    
    # Convert all to numpy arrays
    grids = [np.array(g, dtype=bool) for g in grids]
    
    # Ensure same shape
    shape = grids[0].shape
    if not all(g.shape == shape for g in grids):
        raise ValueError("All grids must have the same shape.")
    
    # Default colors (distinct and clear)
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange'][:n]
    
    # Create a blank base
    plt.figure(figsize=(6, 6))
    
    for i, (grid, color) in enumerate(zip(grids, colors)):
        # Use a masked array to only color True cells
        masked = np.ma.masked_where(~grid, grid)
        plt.imshow(masked, cmap=plt.cm.colors.ListedColormap([color]), alpha=alpha)
    
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def visualize_boolean_grids_on_image(
    image_path,
    grids,
    colors=None,
    alpha=0.4,
    image_alpha=1.0,
    labels=None,
    title="Overlayed Boolean Grids"
):
    """
    Visualize multiple boolean 2D grids overlaid with transparency on top of an image.

    Parameters:
        image_path (str): Path to the background image (e.g. 'background.png').
        grids (list of 2D np.ndarray or lists): Boolean matrices (smaller or equal in resolution to the image).
        colors (list of str): Colors for each grid. Defaults to ['red', 'blue', 'green', 'orange'].
        alpha (float): Transparency of overlay grids (0 = invisible, 1 = opaque).
        image_alpha (float): Transparency of the background image (0 = transparent, 1 = opaque).
        labels (list of str): Legend labels corresponding to each grid (e.g. ["0: left", "1: right", ...]).
        title (str): Title of the plot.
    """
    # Load background image
    img = mpimg.imread(image_path)
    img_h, img_w = img.shape[:2]
    
    # Convert all to boolean arrays
    grids = [np.array(g, dtype=bool) for g in grids]
    
    # Default colors
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange'][:len(grids)]
    
    # Default labels if not provided
    if labels is None:
        labels = [f"{i}: grid {i}" for i in range(len(grids))]
    
    # Upscale each grid to image size (nearest-neighbor)
    upscaled_grids = [
        resize(grid.astype(float), (img_h, img_w), order=0, preserve_range=True).astype(bool)
        for grid in grids
    ]
    
    # Plot setup
    plt.figure(figsize=(6, 6))
    plt.imshow(img, origin='upper', alpha=image_alpha)
    
    # Overlay each grid
    for grid, color in zip(upscaled_grids, colors):
        masked = np.ma.masked_where(~grid, grid)
        plt.imshow(masked, cmap=plt.cm.colors.ListedColormap([color]), alpha=alpha, origin='upper')
    
    # Legend setup
    legend_patches = [Patch(color=c, label=l, alpha=alpha) for c, l in zip(colors, labels)]
    plt.legend(handles=legend_patches, loc='upper right', framealpha=0.8)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()