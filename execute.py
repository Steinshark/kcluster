
import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import svds

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import SparsePCA, IncrementalPCA
from sklearn.cluster import MiniBatchKMeans

from matplotlib import pyplot as plt
from time import time

from colors import *
#path        = r'/mnt/beegfs/m226252/clustering'
#docNYT      = fr'/mnt/beegfs/m226252/clustering/docword.nytimes.txt'
#vocabNYT    = fr'{path}/vocab.nytimes.txt'
docNYT      = fr'newData'
#docNYT	     = fr"docword.nytimes.txt"
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


def tf_calc(csr_matr):
	strs = []
	print(csr_matr.shape[0])
	for i in range(csr_matr.shape[0]):
		strs.append(' '.join(map(str,[i for i,x in enumerate(csr_matr[i].toarray()[0]) if not x == 0])))
	print(strs[:20])


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
    printc(f"Starting: SVD CALC","BLUE")
    k = 150
    printc(f"\tUsing k value of {k}","TAN")

    U, S, Vt = svds(matrix,k=k)
    t3 = time()
    printc(f"\tU: {U.shape}, S: {S.shape}, Vt: {Vt.shape}","TAN")
    printc(f"\tFinished: SVD CALC in  {t3-t2} seconds\n\n","GREEN")

    ############################################################################
    ################### Dimensional Reduction via PCA  #########################
    ############################################################################
    t4 = time()
    redux = 100
    bSize = 50000
    printc(f"Starting PCA","BLUE")
    print(f'{Color.colors["TAN"]}\tPCA reduction to {redux} {Color.colors["END"]}')
    pca = IncrementalPCA(n_components=redux,batch_size=bSize)
    for i in [0,1,2,3,4,5]:
        matrix_transform = pca.fit(matrix)
    err = pca.explained_variance_ratio_
    t5 = time()
    printc(f"\tfinished PCA in {t5-t4} seconds\n\n","GREEN")
    #printc(f"PCA evr: {err}\n\n","GREEN")
    ############################################################################
    ################### Dimensional Reduction via PCA  #########################
    ############################################################################
    t6 = time()
    cluster_sizes = list(range(5,10))
    model = [None for _ in cluster_sizes]
    printc(f"Running KMeans clustering for {cluster_sizes}","BLUE")
    for i,n in enumerate(cluster_sizes):
    	model[i] = MiniBatchKMeans(n_clusters=n, batch_size = bSize)
    	model[i].fit(matrix)
    	printc(f"\tCluster size {i} inertia: {model[i].inertia_}","TAN")
