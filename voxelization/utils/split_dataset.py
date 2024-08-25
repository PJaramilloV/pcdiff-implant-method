import os
import csv
import math
import random
import argparse

data = []
defects = ['broken']
data_dir = 'data/pjaramil/'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='_bottles', help="set dataset name")
parser.add_argument('--format', type=str, default='obj', help="extension file format of original data (default= 'obj')", choices=['obj', 'nrrd'])
opt = parser.parse_args()
dataset = opt.dataset
ex_format = f'.{opt.format}'
voxelization = 'voxelization'

with open(os.path.join(data_dir, dataset, 'train.csv'), 'r', newline='') as file:
    csvreader = csv.reader(file)
    objects = 0
    for row in csvreader:
        split = row[0].split('complete')
        path = split[0]
        file = split[1].split(ex_format)[0][1:] # /file.obj
        data.append(os.path.join(path, voxelization, file))
        objects += 1
    print(f"Found {objects} {dataset} scans")

train = random.sample(data, math.ceil(len(data)*0.70))
test = [elem for elem in data if elem not in train]

os.makedirs(os.path.join(data_dir, dataset, voxelization), exist_ok=True)

# Training set
with open(os.path.join(data_dir, dataset, voxelization,'train.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(train)):
        datapoint = train[i]
        writer.writerow([datapoint])

# Test set
with open(os.path.join(data_dir, dataset, voxelization, 'eval.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(test)):
        datapoint = test[i]
        writer.writerow([datapoint])

print(f"Successfully created training and evaluation split for {dataset}")
