
import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix, csr_matrix, load_npz, save_npz
from scipy.sparse.linalg import svds
import scipy
from json import dumps

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import SparsePCA, IncrementalPCA, TruncatedSVD
from sklearn.cluster import MiniBatchKMeans

from matplotlib import pyplot as plt
from time import time
import sys
from colors import *
path        = r'/mnt/beegfs/m226252/clustering'
if sys.argv[1] == 'hpc':
    docNYT      = fr'/mnt/beegfs/m226252/clustering/docword.nytimes.txt'
elif sys.argv[1] == 'full':
    docNYT	     = fr"docword.nytimes.txt"
else:
    docNYT = 'newData'

vocabNYT    = fr'vocab.nytimes.txt'
if __name__ == "__main__":
    w_file = open("out1",'w')
    w_file.write("\n")
    w_file.close()

def printc(s,color):
    if True:
        w_file = open("out1",'a')
        w_file.write(f"{Color.colors[color]}{s}{Color.colors['END']}\n")
        w_file.close()
    else:
        print(f"{Color.colors[color]}{s}{Color.colors['END']}\n")

def create_csr_matrix(filename,header=3,verbose=False,npzname=None):

    # Import our NYTIMES doc and read the init values
    if not npzname == 'none':
        vocab = {}
        with open(vocabNYT,'r') as vocab_file:
            for i, word in enumerate(vocab_file.readlines()):
                vocab[i] = word
        printc(f"\tBegin {npzname} read via npz","TAN")

        a = load_npz(str(npzname))
        return a, vocab

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

    	matrix = lil_matrix( (rows,cols), dtype = np.float64)


    	# Step through each article and see which word appeared in it
    	t3 = time()
    	docwords = {}
    	printc(f"\tBegin {filename} read of {n_words_total} lines via file IO","TAN")

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

def save_sparse_to_file(matrix,fname):
    save_npz(fname,matrix)

def verbose_read(npz_in_name='', save=False,filename=''):
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    ############################################################################
    #### build our sparce matrix and a dictionary of docID -> words_in_doc  ####
    ############################################################################
    printc(f"Starting: Matrix creation","BLUE")
    t1 = time()
    matrix, docwords = create_csr_matrix(docNYT,header=3,verbose=True,npzname=npz_in_name)
    t2 = time()
    printc(f"\tMatrix created: {docNYT} shape {matrix.shape}","TAN")
    printc(f"\tsize: {matrix.data.size/(1024**2):.2f} MB","TAN")
    printc(f"\tFinished: matrix creation in {t2-t1} seconds\n\n","GREEN")

    if save:
        printc(f"Starting file save of Matrix: {matrix.shape}","BLUE")
        save_sparse_to_file(matrix,filename)
        printc(f"\tsaved/n/n","TAN")

    return matrix, docwords

def verbose_svd_decomp(matrix,n):
    printc(f"Starting: reduction via SVD","BLUE")
    t1 = time()
    printc(f"\ttrying n={n}","TAN")
    tsvd = TruncatedSVD(n_components=n)
    a = tsvd.fit_transform(matrix)
    printc(f"\tmatrix reduced to: {a.shape}","TAN")
    printc(f"\tvar: {tsvd.explained_variance_ratio_.sum(): .4f} in {time()-t1} ","TAN")
    return a

def save_svd_decomp(m_reduced,fname):
    np.save(fname,m_reduced)

def load_svd_decomp(filename):
    return np.load(filename)

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

def run_kmeans_verbose(matrix,move):
    t1 = time()
    bSize = 10000
    printc(f"Starting KMeans","BLUE")
    a = matrix
    models = {}
    for k in [50,100,500,1000,5000,10000]:
        models[k] = {'centers': None, 'd_to_c' : None, 'inertia' : 0}
        t2 = time()
        printc(f"\tStarting k={k} on {a.shape}:","BLUE")
        model = MiniBatchKMeans(n_clusters=k, batch_size = bSize,n_init=3)
        model.fit(a)
        printc(f"\t\tcomputed model k={k} in {time()-t2} seconds:","TAN")
        printc(f"\t\tk={k} inertia: {model.inertia_}","TAN")
        models[k]['inertia'] = model.inertia_
        models[k]["centers"] = model.cluster_centers_
        models[k]['d_to_c'] = model.predict(a)
        printc(f"\t\tpredict finished - writing files","TAN")
        np.save(f"data/{k}_centers",models[k]['centers'])
        np.save(f"data/{k}_d_to_clusters",models[k]['d_to_c'])
        printc(f"\t\tFinished model in {time()-t2} seconds","GREEN")

    return models
    printc(f"\t\tfinished all k clusters in {time()-t1} seconds","TAN")

if not __name__ == "__main__":

    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    ############################################################################
    #### build our sparce matrix and a dictionary of docID -> words_in_doc  ####
    ############################################################################
    printc(f"Starting: Matrix creation","BLUE")
    t1 = time()
    npz_name = input("npzname: ")
    matrix, docwords = create_csr_matrix(docNYT,header=3,verbose=True,npzname=npz_name)
    t2 = time()
    printc(f"\tMatrix created: {docNYT} shape {matrix.shape}","TAN")
    printc(f"\tsize: {matrix.data.size/(1024**2):.2f} MB","TAN")
    printc(f"\tFinished: matrix creation in {t2-t1} seconds\n\n","GREEN")

    printc(f"Starting file save of Matrix: {matrix.shape}","BLUE")
    save_sparse_to_file(matrix)
    printc(f"\tsaved/n/n","TAN")

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
    for n in [int(input("n: " ))]:
        t1 = time()
        printc(f"\ttrying n={n}","TAN")
        tsvd = TruncatedSVD(n_components=n)
        a = tsvd.fit_transform(matrix)
        printc(f"\tmatrix reduced to: {a.shape}","TAN")
        printc(f"\tvar: {tsvd.explained_variance_ratio_.sum(): .4f} in {time()-t1} ","TAN")


    ############################################################################
    ########################## KMeans analysis  ################################
    ############################################################################
    t6 = time()
    bSize = 50000

    cluster_sizes = [100]
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

else:
    n = int(sys.argv[2])
    raw_data_name   =       sys.argv[3]
    saving_raw_data =       sys.argv[4] in ["T",'t']
    loading_svd     =       sys.argv[5] in ['T','t']
    k_start         = int(  sys.argv[6])
    k_end           = int(  sys.argv[7])
    k_inc           = int(  sys.argv[8])



    printc(f"looking for words in: {raw_data_name}","BLUE")
    m,dw = verbose_read(npz_in_name=raw_data_name,save=saving_raw_data,filename='preSVD')

    if loading_svd:
        printc(f"\tloading in precomputed SVD matrix","GREEN")
        m_red = load_svd_decomp(f"decomp_to_{n}.npy")
    else:
        m_red = verbose_svd_decomp(m,n)
        save_svd_decomp(m_red,f"decomp_to_{n}.npy")
    printc(f"\tpost SVD shape: {m_red.shape}\n","BLUE")
    move = np.arange(k_start,k_end,k_inc)
    models = run_kmeans_verbose(m_red,move)



    printc(f"\tStarting kmeans","BLUE")
    t= time()
    doc_to_cluster = model.predict(m_red)
    printc(f"Found clusters: {doc_to_cluster.shape} in {time()-t}","GREEN")
    printc(f"{doc_to_cluster[:2]}","TAN")
    np.save("doc_to_cluster",model.predict(m_red))
