import multiprocessing
import time
import argparse
import numpy as np
import nrrd
import os
import open3d as o3d
from tqdm import tqdm
import sys
import torch
import random

data_dir = 'data/pjaramil/'

parser = argparse.ArgumentParser()
parser.add_argument('--multiprocessing', type=eval, default=True, help="set multiprocessing True/False")
parser.add_argument('--threads', type=int, default=16, help="define the number of threads")
parser.add_argument('--keep_mesh', type=eval, default=False, help="save meshes True/False")
parser.add_argument('--dataset', type=str, default='_bottles', help='directory housing a dataset of complete and broken .obj files')
opt = parser.parse_args()

database = []
multiprocess = opt.multiprocessing
njobs = opt.threads
keep_meshes = opt.keep_mesh
dataset = opt.dataset

# Function to process one file using PyTorch on GPU
def process_one(file_obj, device):
    try:
        complete_pc_filename = file_obj.split('.obj')[0] + '.npy'
        if os.path.exists(complete_pc_filename):
            return

        # Load the TriangleMesh using Open3D
        complete_surf = o3d.io.read_triangle_mesh(file_obj)
        
        # Extract vertices and convert to PyTorch tensor
        vertices = torch.tensor(complete_surf.vertices, dtype=torch.float32, device=device)

        # Sample points using PyTorch on the GPU
        complete_pc = sample_points_poisson_disk(vertices, 400000, device=device)
        
        complete_pc_np = complete_pc.cpu().numpy()
        np.save(complete_pc_filename, complete_pc_np)
    except RuntimeError as e:
        with open('preproc.log', 'a') as f:
            f.write(file_obj + '\n')

def sample_points_poisson_disk(vertices, num_points, radius=0.1, num_attempts=30, device='cuda:0'):
    # Convert radius to squared radius for efficient distance comparison
    radius_sq = radius * radius

    # Initialize an empty list to store the sampled points
    sampled_points = []

    # Initialize a grid to accelerate the search for neighboring points
    grid = {}

    # Convert vertices to PyTorch tensor (if not already)
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32, device=device)

    # Choose a random starting point from the vertices
    start_point = random.choice(vertices.cpu().tolist())

    # Add the starting point to the sampled points
    sampled_points.append(start_point)

    # Initialize a list to keep track of active points
    active_points = [start_point]

    while len(sampled_points) < num_points and active_points:
        # Randomly choose an active point
        current_point = random.choice(active_points)

        found_valid_point = False

        for _ in range(num_attempts):
            # Generate a random point in the annulus (ring) around the current point
            r = radius * (1 + torch.rand(1, device=device))
            theta = 2 * 3.14159265359 * torch.rand(1, device=device)
            x = radius * (2 * torch.rand(1, device=device) - 1)
            y = radius * (2 * torch.rand(1, device=device) - 1)
            offset = torch.tensor([x, y], device=device)

            # Calculate the candidate point
            candidate_point = current_point + offset.cpu().tolist()

            # Check if the candidate point is within the valid range and not too close to existing points
            valid_candidate = (
                candidate_point[0] >= 0 and candidate_point[1] >= 0 and
                candidate_point[0] < 1 and candidate_point[1] < 1
            )
            if valid_candidate:
                valid_candidate = is_valid_point(candidate_point, radius_sq, sampled_points, grid, device=device)

            if valid_candidate:
                found_valid_point = True
                sampled_points.append(candidate_point)
                active_points.append(candidate_point)
                grid_key = grid_key_from_point(candidate_point, radius)
                grid[grid_key] = candidate_point
                break

        if not found_valid_point:
            active_points.remove(current_point)

    # Convert the sampled points list to a PyTorch tensor
    sampled_points = torch.tensor(sampled_points, dtype=torch.float32, device=device)

    return sampled_points

def is_valid_point(point, radius_sq, points, grid, device='cuda:0'):
    point = torch.tensor(point, dtype=torch.float32, device=device)  # Convert point to a PyTorch tensor
    for existing_point in points:
        existing_point = torch.tensor(existing_point, dtype=torch.float32, device=device)  # Convert existing_point to a PyTorch tensor
        if torch.sum((point - existing_point) ** 2) < radius_sq:
            return False

    grid_key = grid_key_from_point(point)
    for neighbor_key in get_neighbor_keys(grid_key):
        if neighbor_key in grid:
            neighbor = grid[neighbor_key]
            neighbor = torch.tensor(neighbor, dtype=torch.float32, device=device)  # Convert neighbor to a PyTorch tensor
            if torch.sum((point - neighbor) ** 2) < radius_sq:
                return False

    return True

def grid_key_from_point(point, radius=0.1):
    return tuple((point / radius).to(torch.int).tolist())

def get_neighbor_keys(key):
    x, y = key
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
            (x - 1, y), (x, y), (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

def main():
    directory = os.path.join(data_dir, dataset)
    print(directory)
    counter = 0
    for root, dirs, files in os.walk(directory, topdown=False):
        for filename in files:
            if filename.endswith('.obj'):
                counter += 1
                if not os.path.exists(os.path.join(root, filename.split('obj')[0] + 'npy')):
                    datapoint = os.path.join(root, filename)
                    database.append(datapoint)
    print(f"Found {len(database)}/{counter} files unprocessed")
    
    if multiprocess:
        # Get the list of available GPU devices
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        print(f'Available GPUs: {devices}')
        
        with multiprocessing.Pool(njobs) as pool:
            try:
                for i, obj in enumerate(tqdm(database)):
                    process_one(obj, devices[i % len(devices)])  # Cycle through available GPUs
            except KeyboardInterrupt:
                exit()
    else:
        for obj in tqdm(database):
            process_one(obj, 'cuda:0')  # Use the first GPU

if __name__ == "__main__":
    print(f'Preprocess {dataset} dataset...')
    t_start = time.time()
    main()
    t_end = time.time()
    print('Done. Total processing time: ', t_end - t_start)
