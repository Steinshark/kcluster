
import numpy as np
import pandas as pd
from scipy.sparse import *
from sklearn.decomposition import PCA

path        = r'/mnt/beegfs/m226252/clustering'
docNYT      = fr'{path}/docword.nytimes.txt'
vocabNYT    = fr'{path}/vocab.nytimes.txt'

def read_data_to_df(filename,header=3):
	with open(filename,'r') as file:
		n_articles      = int(file.readline())
		n_words         = int(file.readline())
		n_words_total   = int(file.readline())
		print(f"parsing dataset of len: {n_words_total}")
		print(f"rows: {n_articles}, cols:{n_words}")
		vocab = {}
		with open(vocabNYT,'r') as vocab_file:
			for i, word in enumerate(vocab_file.readlines()):
				vocab[i] = word

	# define the size of the dataset we will build
		rows = n_articles
		cols = n_words
		matrix = lil_matrix((rows,cols), dtype = np.float64)
		i = 0
		for line in file.readlines():
			doc, word, count = line.split(' ')
			matrix[int(doc),int(word)] = int(count)
			if i % 1000000 == 0:
				print(f"{i} ways through")
			i += 1
		print(matrix.getrow())



read_data_to_df(docNYT,header=3)
