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

import cupy



def load_svd_decomp(filename):
    return np.load(filename,)


if __name__ == "__main__":
    fname = f"decomp_to_3500.npy"

    print("loading  docs")
    documents = load_svd_decomp(fname)
    print(f"shape docs: {documents.shape}")

    print("done")
    doc_to_cluster = {None for _ in range(documents.shape[0])}

    centers = np.load('centers.npy')
    print(f"shape centers: {centers.shape}")
    doc_num = 0
    for doc in documents:
        min_cluster = None
        min_dist    = 1000000

        for cluster in centers:
             dist = np.linalg.norm(cluster-doc)
             if dist < min_dist:
                 min_dist = dist
                 min_cluster = cluster
        doc_to_cluster[doc_num] = min_cluster
        doc_num += 1
    f = open("cluster_map",'w')
    f.write(dumps(doc_to_cluster))
    f.close()
