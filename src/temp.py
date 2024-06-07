from utils import pickle_load

import numpy as np

data = pickle_load("../data/pickle/undisturbed_masks.pkl")

print(len(data["06"]["group"].keys()))
print(len(data["06"]["non_group"].keys()))
print(len(data["08"]["group"].keys()))
print(len(data["08"]["non_group"].keys()))
