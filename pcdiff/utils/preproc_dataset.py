import multiprocessing
from tqdm import tqdm
import open3d as o3d
import numpy as np
import argparse
import mcubes
import time
import nrrd
import os



data_dir = 'data/pjaramil/'

parser = argparse.ArgumentParser()
parser.add_argument('--multiprocessing', type=eval, default=True, help="set multiprocessing True/False")
parser.add_argument('--threads', type=int, default=14, help="define number of threads")
parser.add_argument('--keep_mesh', type=eval, default=False, help="save meshes True/False")
parser.add_argument('--dataset', type=str, default='', help='directory housing a dataset of complete and broken .obj files')
parser.add_argument('--datadir', type=str, default='', help='base directory of the dataset, use to override hardcoded variable') 
parser.add_argument('--points', type=int, default=200000, help='Amount of samples created in the Poisson Disk sampling')
parser.add_argument('--format', type=str, default='obj', help="extension file format of original data (default= 'obj')", choices=['obj', 'stl', 'off'])
parser.add_argument('--center', type=str, default='xz', help="axis' along which to center the object, unselected axis are floored", choices=['','x', 'y', 'z', 'xy', 'yz', 'xz', 'xyz'])
parser.add_argument('--offset', type=float, default=0, help='proportion of available space from which to lift the object on a floored dimension after rescaling [0,1]')
parser.add_argument('--size', type=int, default=512, help='length of the cubic space on which the object shall reside')
parser.add_argument('--forced' , type=eval, default=False, help='De do all existing preprocessed objects found')
opt = parser.parse_args()

multiprocess = opt.multiprocessing
keep_meshes = opt.keep_mesh
centered_axis = opt.center
override = opt.forced
dataset = opt.dataset
points = opt.points
offset = opt.offset
njobs = opt.threads
size = opt.size

if opt.datadir:
    data_dir = opt.datadir
ex_format = f'.{opt.format}'
database = []

if offset > 1 or offset < 0:
    raise ValueError('Flooring offset brings object partially or completely out of the cubic space')


def process_one(file_obj):
    # ---------------------------------------------
    # Create point clouds from these surface meshes
    # ---------------------------------------------
    size = 512
    try:
        filename, ext = os.path.splitext(file_obj)
        complete_pc_filename = filename + '.npy'
        if os.path.exists(complete_pc_filename) and (not override):
            return
        if ext == '.nrrd':
            file_nrrd_obj = file_obj
            file_obj = f'{filename}.obj'
            if not os.path.exists(file_obj):
                scan, _ = nrrd.read(file_nrrd_obj)
                shape_vrt, shape_tri = mcubes.marching_cubes(scan,0)
                mcubes.export_obj(shape_vrt, shape_tri, file_obj)

        complete_surf = o3d.io.read_triangle_mesh(file_obj)
        complete_pc = complete_surf.sample_points_poisson_disk(400000)
        complete_pc_np = np.asarray(complete_pc.points)

        # normalize pc
        max_val = np.max(complete_pc_np)
        min_val = np.min(complete_pc_np)
        complete_pc_np = ((complete_pc_np - min_val) / (max_val - min_val)) *(0.95 * size)

        # center pointcloud on selected axis' with others' floored
        min_x, max_x = np.min(complete_pc_np[:, 0]), np.max(complete_pc_np[:, 0])
        min_y, max_y = np.min(complete_pc_np[:, 1]), np.max(complete_pc_np[:, 1])
        min_z, max_z= np.min(complete_pc_np[:, 2]),  np.max(complete_pc_np[:, 2])
        
        empty_x = (size - max_x + min_x)
        empty_y = (size - max_y + min_y)
        empty_z = (size - max_z + min_z)

        # (side pad, floor dist) = (available space, drop) if centered, else (don't move, drop then raise by offset of available space) 
        pad_x, low_x = (empty_x, min_x) if 'x' in centered_axis else (0, min_x - offset*empty_x)
        pad_y, low_y = (empty_y, min_y) if 'y' in centered_axis else (0, min_y - offset*empty_y)
        pad_z, low_z = (empty_z, min_z) if 'z' in centered_axis else (0, min_z - offset*empty_z)

        complete_pc_np += np.array((pad_x/2 - low_x, pad_y/2 - low_y, pad_z/2 - low_z))
        np.save(complete_pc_filename, complete_pc_np)
    except RuntimeError as e:
        with open('preproc.log','a') as f:
            f.write(file_obj+'\n')

def main():
    directory =  os.path.join(data_dir, dataset)
    print(directory)
    # Gather available data
    counter = 0
    for root, dirs, files in os.walk(directory, topdown=False):
        for filename in files:
            if filename.endswith(ex_format):
                counter += 1
                if not os.path.exists(os.path.join(root, filename.split(ex_format)[0] + '.npy') ) or override:
                    datapoint = ''
                    datapoint= os.path.join(root, filename)
                    database.append(datapoint)
    print(f"Found {len(database) }/{counter} files unprocessed")
    if multiprocess:
        pool = multiprocessing.Pool(njobs)
        try:
            for _ in tqdm(pool.imap_unordered(process_one, database), total=len(database)):
                pass
        except KeyboardInterrupt:
            exit()
        pool.close()

    else:
        for obj in tqdm(database):
            process_one(obj)
    log_name = dataset.split('/')[-1]
    log_file = f'preproc{log_name}.log'
    with open(os.path.join(directory, log_file),'w') as log:
        for arg, val in sorted(vars(opt).items()):
            log.write(f"Argument {arg}:{val}\n")


if __name__ == "__main__":
    print(f'Preprocess {dataset} dataset ...')
    t_start = time.time()
    main()
    t_end = time.time()
    print('Done. Total processing time: ', t_end - t_start)
