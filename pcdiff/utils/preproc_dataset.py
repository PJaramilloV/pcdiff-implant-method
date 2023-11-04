import multiprocessing
import time
import argparse
import numpy as np
import nrrd
import os
import open3d as o3d
import mcubes
from tqdm import tqdm
import sys

data_dir = 'data/pjaramil/'





parser = argparse.ArgumentParser()
parser.add_argument('--multiprocessing', type=eval, default=True, help="set multiprocessing True/False")
parser.add_argument('--threads', type=int, default=8, help="define number of threads")
parser.add_argument('--keep_mesh', type=eval, default=False, help="save meshes True/False")
parser.add_argument('--dataset', type=str, default='_bottles', help='directory housing a dataset of complete and broken .obj files') 
opt = parser.parse_args()

database = []
multiprocess = opt.multiprocessing
njobs = opt.threads
keep_meshes = opt.keep_mesh
dataset = opt.dataset

    

def process_one(file_obj):
    # ---------------------------------------------
    # Create point clouds from these surface meshes
    # ---------------------------------------------
    try:
        complete_pc_filename = file_obj.split('.obj')[0] + '.npy'
        if os.path.exists(complete_pc_filename):
            return
        complete_surf = o3d.io.read_triangle_mesh(file_obj)
        complete_pc = complete_surf.sample_points_poisson_disk(400000)
        complete_pc_np = np.asarray(complete_pc.points)
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
        for filename in files[10:]:
            if filename.endswith('.obj'):
                counter += 1
                if not os.path.exists(os.path.join(root, filename.split('obj')[0] + 'npy')):
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


if __name__ == "__main__":
    print(f'Preprocess {dataset} dataset ...')
    t_start = time.time()
    main()
    t_end = time.time()
    print('Done. Total processing time: ', t_end - t_start)
