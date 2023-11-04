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

# Path to the complete_skull folder of SkullBreak dataset
dataset = '_bottles'


parser = argparse.ArgumentParser()
parser.add_argument('--multiprocessing', type=eval, default=True, help="set multiprocessing True/False")
parser.add_argument('--threads', type=int, default=8, help="define number of threads")
parser.add_argument('--keep_mesh', type=eval, default=False, help="save meshes True/False")
opt = parser.parse_args()

database = []
multiprocess = opt.multiprocessing
njobs = opt.threads
keep_meshes = opt.keep_mesh


def process_one(datapoint):
    process_file(datapoint['complete'])
    process_file(datapoint['broken'])
    

def process_file(file_obj):
    # ---------------------------------------------
    # Create point clouds from these surface meshes
    # ---------------------------------------------
    complete_surf = o3d.io.read_triangle_mesh(file_obj)
    complete_pc = complete_surf.sample_points_poisson_disk(400000)
    complete_pc_np = np.asarray(complete_pc.points)
    complete_pc_filename = file_obj.split('.obj')[0] + '.npy'
    np.save(complete_pc_filename, complete_pc_np)


def main():
    directory =  os.path.join(data_dir, dataset)
    print(directory)
    # Gather available data
    for root, dirs, files in os.walk(directory, topdown=False):
        for filename in files:
            if filename.endswith('.obj'):
                datapoint = {}
                parent_dir = root.split('/complete')[0]
                datapoint['broken'] = os.path.join(parent_dir, 'broken', filename)
                datapoint['complete'] = os.path.join(root, filename)
                database.append(datapoint)

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
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    print(f'Preprocess {dataset} dataset ...')
    t_start = time.time()
    main()
    t_end = time.time()
    print('Done. Total processing time: ', t_end - t_start)
