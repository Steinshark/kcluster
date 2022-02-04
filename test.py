
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
docNYT      = fr'newData'
#docNYT	     = fr"docword.nytimes.txt"
vocabNYT    = fr'vocab.nytimes.txt'

def create_csr_matrix2(filename,header=3,verbose=False):

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
		for line in file.read().splitlines():
			doc, word, count = line.split(' ')
			#input(f"doc: {doc}, word: {word}, count: {count}")
			matrix[int(doc),int(word)] = int(count)
			try:
				docwords[int(doc)].append(count)
			except KeyError:
				docwords[int(doc)] = [int(count)]

	return matrix.tocsr(), docwords

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



if __name__ == "__main__":
    n = 10
    times_1 = []
    times_2 = []
    y = range(n)
    for i in range(n):
        t1 = time()
        c,d = create_csr_matrix2(docNYT,header=3,verbose=False)
        t2 = time()
        times_2.append(t2-t1)

        c,d = create_csr_matrix(docNYT,header=3,verbose=False)
        times_1.append(time()-t2)

    plt.scatter(y,times_1,label="iter")
    plt.scatter(y,times_2,label="splitlines")
    print(f"iter: {np.mean(times_1)}, splitlines: {np.mean(times_2)}")
    plt.show()
