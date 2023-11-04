import os
import csv
import math
import glob
import random
import argparse

data_dir = 'data/pjaramil/'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='_bottles', help="set dataset name")
opt = parser.parse_args()
dataset = opt.dataset

completes = sorted(glob.glob(os.path.join(data_dir,  dataset, '*.obj')))
print(f"Found {len(completes)} {dataset} scans")
if (len(completes) == 0):
    print("Error: No scans found in the specified directory. Please check the dataset path.")
    exit(0)

with open(os.path.join(data_dir,  dataset+'.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(completes)):
        complete = completes[i]
        writer.writerow([complete])

train = random.sample(completes, math.ceil(len(completes)*0.75))
test = [elem for elem in completes if elem not in train]

# Training set
with open(os.path.join(data_dir,  dataset, 'train.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(train)):
        complete = train[i]
        writer.writerow([complete])

# Test set
with open(os.path.join(data_dir,  dataset, 'test.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(test)):
        complete = test[i]
        writer.writerow([complete])

print(f"Successfully created training and test split for {dataset}")
