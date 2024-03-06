import os
import argparse
import multiprocessing
import numpy as np
from tqdm import tqdm
import nrrd
import time
import csv
import trimesh
import open3d as o3d
import scipy.ndimage as sci

parser = argparse.ArgumentParser()
parser.add_argument('--multiprocessing', type=eval, default=True, help="set multiprocessing True/False")
parser.add_argument('--threads', type=int, default=8, help="define number of threads")
parser.add_argument('--num_points', type=int, default=30720, help="number of points the point cloud should contain")
parser.add_argument('--num_nn', type=int, default=3072, help="number of points that represent the implant")
parser.add_argument('--dataset', type=str, default='_bottles', required=True, help="set dataset name")
parser.add_argument('--path', type=str, default='data/pjaramil/', help="set dataset path")
parser.add_argument('--format', type=str, default='obj', help="extension file format of original data (default= 'obj')", choices=['obj', 'nrrd'])
opt = parser.parse_args()

multiprocess = opt.multiprocessing
njobs = opt.threads
save_pointcloud = True
save_psr_field = True
num_points = opt.num_points
num_nn = opt.num_nn
ex_format = f'.{opt.format}'
padding = 1.2
mesh_factor = 1.1


def array2voxel(voxel_array):
    """
    convert a to a fixed size array to voxel_grid_index array
    (voxel_size*voxel_size*voxel_size)->(N*3)

    :input voxel_array: array with shape(voxel_size*voxel_size*voxel_size),the grid_index in
    :return grid_index_array: get from o3d.voxel_grid.get_voxels()
    """
    x, y, z = np.where(voxel_array == 1)
    index_voxel = np.vstack((x, y, z))
    grid_index_array = index_voxel.T
    return grid_index_array

def obj_to_numpy(file):
    mesh = trimesh.load(file)

    # Get the vertices of the mesh
    vertices = mesh.vertices

    # Choose the dimensions of the voxel grid
    dims = (512, 512, 512)
    dims = np.array(dims)

    # Calculate the voxel grid
    voxel_grid = np.zeros(dims, dtype=bool)

    # Normalize vertices to be in the range [0, 1] within the voxel grid
    normalized_vertices = (vertices - mesh.bounds[0]) / np.max(mesh.extents)

    # Calculate voxel indices
    indices = (normalized_vertices * (dims - 1)).astype(int)

    # Set the corresponding voxels to True
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    return voxel_grid

def obj_to_vox(file):
    mesh:trimesh.Trimesh = trimesh.load(file)
    grid_size = 512
    grid_size_arr = np.array((grid_size, grid_size, grid_size))

    # voxelize mesh into matrix
    max_size = np.max(mesh.extents)
    scaling = max_size/(grid_size - 6)

    vox:trimesh.voxel.VoxelGrid = mesh.voxelized(scaling)
    voxel_grid = vox.matrix
    voxel_grid_shape = np.array(voxel_grid.shape)

    # pad to final size
    pad_x, pad_y, pad_z = (grid_size_arr - voxel_grid_shape) // 2
    fix_x, fix_y, fix_z = grid_size_arr - voxel_grid_shape - 2*np.array((pad_x, pad_y, pad_z))
    vox_out = np.pad(voxel_grid, 
                     ((pad_x, pad_x+fix_x),(pad_y, pad_y+fix_y),(pad_z, pad_z+fix_z)),
                     mode='constant', constant_values=0)
    return vox_out


def process_one(obj):
    pc_np   = np.load(obj['broken'])  # Defective pc
    pc_d_np = np.load(obj['repair'])  # Implant pc
    if ex_format == '.obj':            # Gt vox
        gt_vox = obj_to_vox(obj['gt_vox'])  
    elif ex_format == '.nrrd':
        gt_vox, _ = nrrd.read(obj['gt_vox'])
    else:
        raise TypeError(f"Unsupported data type: {ex_format}")

    # Downsample point clouds, broken and desired repair
    num_pc = pc_np.shape[0]
    idx_pc = np.random.randint(0, num_pc, num_points - num_nn)
    pc_np = pc_np[idx_pc, :]

    num_pc_d = pc_d_np.shape[0]
    idx_pc_d = np.random.randint(0, num_pc_d, num_nn)
    pc_d_np = pc_d_np[idx_pc_d, :]

    points = np.concatenate((pc_np, pc_d_np), axis=0)  # Pc complete (defective skull + implant)
    
    # normalize pc
    max_val = np.max(points)
    min_val = np.min(points)
    points = ((points - min_val) / (max_val - min_val)) *0.95

    # center pointcloud on x-y with z floored
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    min_z= np.min(points[:, 2])
    
    pad_x = 1 - max_x + min_x
    pad_y = 1 - max_y + min_y
    points += np.array((pad_x/2 - min_x, pad_y/2 - min_y, -min_z))

    # create voxel representation of gt
    vox = np.ones((512, 512, 512), dtype=bool) * 0.5
    vox[gt_vox > 0] = -0.5
    vox = vox.astype(np.float32)

    # save
    path = obj['broken'].split('/broken')[0]
    obj_name = obj['gt_vox'].split('/')[-1].split(ex_format)[0]
    name_vox = os.path.join(path, 'voxelization', obj_name + '_vox.npz')
    name_points = os.path.join(path, 'voxelization', obj_name + '_pc.npz')

    if save_pointcloud:
        np.savez_compressed(name_points, points=points)
    if save_psr_field:
        np.savez_compressed(name_vox, psr=vox)


def main():

    dataset_path = opt.path
    dataset_name = opt.dataset

    print('---------------------------------------')
    print(f'Processing {dataset_name} dataset')
    print('---------------------------------------')
    
    dataset_csv = os.path.join(dataset_path, dataset_name , f'{dataset_name}.csv')
    database = []

    vox_dir = os.path.join(dataset_path, dataset_name, 'voxelization')
    if not os.path.isdir(vox_dir):
        os.makedirs(vox_dir)

    with open(dataset_csv, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            split = row[0].split('complete')
            path = split[0]
            file = split[1].split(ex_format)[0][1:] # -->/file<--.obj
            datapoint = dict()
            datapoint['broken'] = os.path.join( path , 'broken' , file + '.npy')
            datapoint['repair'] = os.path.join( path , 'repair' , file + '.npy')
            datapoint['gt_vox'] = row[0]
            database.append(datapoint)

    if multiprocess:
        # multiprocessing.set_start_method('spawn', force=True)
        pool = multiprocessing.Pool(njobs)
        try:
            for _ in tqdm(pool.imap_unordered(process_one, database), total=len(database)):
                pass
            # pool.map_async(process_one, obj_list).get()
        except KeyboardInterrupt:
            # Allow ^C to interrupt from any thread.
            exit()
        pool.close()
    else:
        for obj in tqdm(database):
            process_one(obj)

    print('Done Processing')


if __name__ == "__main__":
    t_start = time.time()
    main()
    t_end = time.time()
    print('Total processing time: ', t_end - t_start)
