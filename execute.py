
import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import svds

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import SparsePCA, IncrementalPCA, TruncatedSVD
from sklearn.cluster import MiniBatchKMeans

from matplotlib import pyplot as plt
from time import time
import sys
from colors import *
#path        = r'/mnt/beegfs/m226252/clustering'
try:
    if sys.argv[1] == 'hpc':
        docNYT      = fr'/mnt/beegfs/m226252/clustering/docword.nytimes.txt'
except:
    docNYT	     = fr"docword.nytimes.txt"

#vocabNYT    = fr'{path}/vocab.nytimes.txt'
#docNYT      = fr'newData'
vocabNYT    = fr'vocab.nytimes.txt'

def printc(s,color):
    print(f"{Color.colors[color]}{s}{Color.colors['END']}")

def create_csr_matrix(filename,header=3,verbose=False):

    # Import our NYTIMES doc and read the init values
    with open(filename,'r') as file:
    	n_articles      = int(file.readline())
    	n_words         = int(file.readline())
    	n_words_total   = int(file.readline())


    	# Create a dictionary so we can map word_ID to actual text in the NYTIMES doc
    	t1 = time()
    	vocab = {}
    	with open(vocabNYT,'r') as vocab_file:
    		for i, word in enumerate(vocab_file.readlines()):
    			vocab[i] = word
    	t2 = time()
    	printc(f"\tFinished vocab read of {len(vocab)} words in {t2-t1} seconds","TAN")
    	# define the size of the dataset we will build
    	rows = n_articles	+	1
    	cols = n_words		+	1


    	# initialize an lil matrix (faster to fill)
    	matrix = lil_matrix((rows,cols), dtype = np.float64)


    	# Step through each article and see which word appeared in it
    	t3 = time()
    	docwords = {}
    	printc(f"\tBegin docword read of {n_words_total} lines","TAN")

    	for line in file:
    		doc, word, count = line.split(' ')
    		#input(f"doc: {doc}, word: {word}, count: {count}")
    		matrix[int(doc),int(word)] = int(count)
    		try:
    			docwords[int(doc)].append(count)
    		except KeyError:
    			docwords[int(doc)] = [int(count)]
    	t4 = time()
    	printc(f"\tFinished docword read of {n_words_total} words in {t4-t3} seconds","TAN")

    return matrix.tocsr(), docwords


def svd_calc(sparse_matrix,k=150,verbose=False):
    if verbose:
        printc(f"Starting: SVD CALC","BLUE")
        printc(f"\tk value: {k}\n\tmatrix dim: {sparse_matrix.shape}\n\tmatrix size: {matrix.data.size/(1024**2):.2f} MB","TAN")

    U, S, Vt = svds(matrix,k=k)

    if verbose:
        t3 = time()
        printc(f"\tU: {U.shape}, S: {S.shape}, Vt: {Vt.shape}","TAN")
        printc(f"\tFinished: SVD CALC in  {t3-t2} seconds\n\n","GREEN")
    return U,S,Vt


if __name__ == "__main__":

    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    ############################################################################
    #### build our sparce matrix and a dictionary of docID -> words_in_doc  ####
    ############################################################################
    printc(f"Starting: Matrix creation","BLUE")
    t1 = time()
    matrix, docwords = create_csr_matrix(docNYT,header=3,verbose=True)
    t2 = time()
    printc(f"\tMatrix created: {matrix.shape}","TAN")
    printc(f"\tsize: {matrix.data.size/(1024**2):.2f} MB","TAN")
    printc(f"\tFinished: matrix creation in {t2-t1} seconds\n\n","GREEN")


    ############################################################################
    ################## Calculate the SVD for the matrix ########################
    ############################################################################

    #U, S, Vt = svd_calc(matrix,k=50,verbose=True)

    ############################################################################
    ################### Dimensional Reduction via PCA  #########################
    ############################################################################
    printc(f"Starting: reduction via SVD","BLUE")
    per_var = []
    n_val   = []
    t_comp  = []
    for n in [2000]:
        t1 = time()
        printc(f"\ttrying n={n}","TAN")
        tsvd = TruncatedSVD(n_components=n)
        a = tsvd.fit_transform(matrix)
        printc(f"\tmatrix reduced to: {a.shape}","TAN")
        printc(f"\tvar: {tsvd.explained_variance_ratio_.sum(): .4f} in {time()-t1} ","TAN")
    return
    ############################################################################
    ########################## KMeans analysis  ################################
    ############################################################################
    t6 = time()
    bSize = 50000

    cluster_sizes = [1]
    model = [None for _ in cluster_sizes]
    printc(f"Starting KMeans","BLUE")
    printc(f"\tRunning k-vals of: {cluster_sizes}","BLUE")
    for i,n in enumerate(cluster_sizes):
        t1 = time()
        printc(f"\t\tStarting {i}:","TAN")
        model[i] = MiniBatchKMeans(n_clusters=n, batch_size = bSize,n_init=9)
        model[i].fit(a)
        printc(f"\t\tFinished {i} in {time()-t1} seconds:","TAN")

        printc(f"\t{n} clusters inertia: {model[i].inertia_}","TAN")


    ############################################################################
    ########################### Truncated SVD  #################################
    ############################################################################
