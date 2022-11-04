import pandas as pd
import os
import numpy as np
import torch


# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("DEVICE: {}".format(DEVICE))

# replace with location of folder containing "gtFine" and "leftImg8bit"
CITYSCAPES_DATASET = "/media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/segmentation/datasets/cityscapes"

best_accuracy = 0

LEARNING_RATE  = 1e-6
TRAIN_EPOCHS = 50
N_CLASSES = 19
BATCH_SIZE = 8
NUM_WORKERS = 4