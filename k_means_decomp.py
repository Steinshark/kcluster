import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix, csr_matrix, load_npz, save_npz
from scipy.sparse.linalg import svds
import scipy
from json import dumps, loads


from matplotlib import pyplot as plt
from time import time
import sys
from colors import *

def printc(s,color):
    print(f"{Color.colors[color]}{s}{Color.colors['END']}\n")

def load_svd_decomp(filename):
    return np.load(filename)


if __name__ == "__main__":
    clusters = np.load('doc_to_cluster.npy')
    printc(f"loaded shape {clusters.shape}","GREEN")

    doc_to_cluster = {i : clusters[i] for i in range(300000)}

    for k in doc_to_cluster:
        printc(f"{k} -> {doc_to_cluster[k]}","TAN")
        input()
