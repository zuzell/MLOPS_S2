import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.data.make_dataset import MNISTdata
import torch

#dataset = MNIST(...)
test_data = torch.load("../data/processed/test.pth")
train_data = torch.load("../data/processed/train.pth")

N_train = 25000
N_test=5000


assert len(train_data) == N_train  and len(test_data) ==N_test, f"Size of the dataset is not valid"


assert all ([image.shape == torch.Size([1, 28, 28]) for image, label in train_data]), "Data has wrong size"

assert all ([label != None for image, label in train_data]), "Empty label"
