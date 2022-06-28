import numpy as np
import imageio as iio
import os
from joblib import Parallel, delayed
import sys
import matplotlib.pyplot as plt

def pairwise_diff(prev, curr, prev_index):
    x = iio.imread(prev, as_gray = True)
    y = iio.imread(curr, as_gray = True)
    diff = (y - x).flatten()
    res = np.array([diff.sum(), diff.mean(), diff.std()])

    return [prev_index, res]

if len(sys.argv) != 2:
    quit("Need to provide a path to the images.")

# Set up some vars.
files = list(filter(lambda x: x.endswith(".png"), os.listdir(sys.argv[1])))
files.sort(key = lambda f: int(f.split(".")[0].split("_")[1]))
digits = 5
window = 5
length = len(files)

out_array = np.zeros(shape = (length - 1, 3))

out = Parallel(n_jobs = -1, verbose = 50)(
        delayed(pairwise_diff)
        (files[i], files[i + 1], i) for i in range(length - 1))

for i, res in out:
    out_array[i] = res

np.save("out_array.npy", out_array)
