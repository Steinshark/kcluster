
import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import svds

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import SparsePCA

from matplotlib import pyplot as plt
from time import time

from colors import *
#path        = r'/mnt/beegfs/m226252/clustering'
#docNYT      = fr'{path}/docword.nytimes.txt'
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

		# Update us on whats going on
		if verbose:
			print(f"parsing dataset of len: {n_words_total}")
			print(f"rows: {n_articles}, cols:{n_words}")

		# Create a dictionary so we can map word_ID to actual text in the NYTIMES doc
		vocab = {}
		with open(vocabNYT,'r') as vocab_file:
			for i, word in enumerate(vocab_file.readlines()):
				vocab[i] = word

		# define the size of the dataset we will build
		rows = n_articles	+	1
		cols = n_words		+	1

		# initialize an lil matrix (faster to fill)
		matrix = lil_matrix((rows,cols), dtype = np.float64)


		# Step through each article and see which word appeared in it
		docwords = {}
		for line in file:
			doc, word, count = line.split(' ')
			#input(f"doc: {doc}, word: {word}, count: {count}")
			matrix[int(doc),int(word)] = int(count)
			try:
				docwords[int(doc)].append(count)
			except KeyError:
				docwords[int(doc)] = [int(count)]

	return matrix.tocsr(), docwords


def tf_calc(csr_matr):
	strs = []
	print(csr_matr.shape[0])
	for i in range(csr_matr.shape[0]):
		strs.append(' '.join(map(str,[i for i,x in enumerate(csr_matr[i].toarray()[0]) if not x == 0])))
	print(strs[:20])


def SVD_decomp(csr_matr,k=100):
	return svds(sparse_matrix,k=150)


if __name__ == "__main__":
	############################################################################
	#### build our sparce matrix and a dictionary of docID -> words_in_doc  ####
	############################################################################
	t1 = time()
	matrix, docwords = create_csr_matrix(docNYT,header=3,verbose=True)
	t2 = time()
	printc(f"finished building matrix {matrix.shape} in {t2-t1} seconds","GREEN")


	############################################################################
	################## Calculate the SVD for the matrix ########################
	############################################################################
	printc(f"Starting SVD CALC","BLUE")
	U, S, Vt = svds(matrix,k=100)
	t3 = time()
	printc(f"finished building SVD in {t3-t2} seconds","GREEN")
	printc(f"U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}","BLUE")

	############################################################################
	################### Dimensional Reduction via PCA  #########################
	############################################################################
	t4 = time()
	redux = 100
	print(f'{Color.colors["TAN"]}PCA reduction size: {redux} {Color.colors["END"]}')
	printc(f"Beginning PCA...","BLUE")
	pca = SparsePCA(n_components=redux)
	matrix_transform = pca.fit_transform(matrix)
	err = pca.explained_variance_ratio_
	t5 = time()
	printc(f"finished PCA in {t5-t4} seconds","GREEN")

	############################################################################
	################### Dimensional Reduction via PCA  #########################
	############################################################################
	printc(f"RUnning KMeans clustering on {type()}")
