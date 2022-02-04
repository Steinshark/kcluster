
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.decomposition import PCA
from copy import copy
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
from time import time
#path        = r'/mnt/beegfs/m226252/clustering'
#docNYT      = fr'{path}/docword.nytimes.txt'
#vocabNYT    = fr'{path}/vocab.nytimes.txt'
#docNYT      = fr'newData'
docNYT	     = fr"docword.nytimes.txt"
vocabNYT    = fr'vocab.nytimes.txt'

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
	# build our sparce matrix and a dictionary of docID -> words_in_doc
	t1 = time()
	sparse_matrix, docwords = create_csr_matrix(docNYT,header=3,verbose=True)
	t2 = time()
	print(f"finished building matrix {sparse_matrix.shape} in {t2-t1} seconds")

	# Calculate the SVD for the matrix
	U, S, Vt = svds(sparse_matrix,k=100)
	print(f"finished building SVD in {time()-t2} seconds")
