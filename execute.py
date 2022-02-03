
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.decomposition import PCA
from numpy.linalg import svd, LinAlgError
from copy import copy

path        = r'/mnt/beegfs/m226252/clustering'
docNYT      = fr'{path}/docword.nytimes.txt'
vocabNYT    = fr'{path}/vocab.nytimes.txt'

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
		for i, line in enumerate(file.readlines()):
			doc, word, count = line.split(' ')
			matrix[int(doc),int(word)] = int(count)

			# Update
			if verbose and i % 1000000 == 0:
				print(f"{i} ways through")

		return csr_matrix(matrix)

if __name__ == "__main__":
	# build our sparce matrix
	sparse_matrix = create_csr_matrix(docNYT,header=3,verbose=True)

	# Try to get the SVD directly
	try:
		U,S,Vh = svd(sparse_matrix)
		sPrime = S[0:-1:100].copy()

		x_sPrime = np.arrange(sPrime.size)

		plt.plot(x_sPrime,sPrime)
		plt.show() 
	except LinAlgError as l:
		print(l)
	except Exception as e:
		print(e)
