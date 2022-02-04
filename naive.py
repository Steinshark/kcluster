
import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import svd

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from time import time

from colors import *
#path        = r'/mnt/beegfs/m226252/clustering'
#docNYT      = fr'{path}/docword.nytimes.txt'
#vocabNYT    = fr'{path}/vocab.nytimes.txt'
#docNYT      = fr'newData'
docNYT	     = fr"docword.nytimes.txt"
vocabNYT    = fr'vocab.nytimes.txt'

def printc(s,color):
    print(f"{Color.colors[color]}{s}{Color.colors['END']}")

def create_matrix(filename,header=3,verbose=False):

    # Import our NYTIMES doc and read the init values
    with open(filename,'r') as file:
        n_articles      = int(file.readline())
        n_words         = int(file.readline())
        n_words_total   = int(file.readline())

        # Create a dictionary so we can map word_ID to actual text in the NYTIMES doc
        vocab = {}
        with open(vocabNYT,'r') as vocab_file:
            for i, word in enumerate(vocab_file.readlines()):
                vocab[i] = word

                # define the size of the dataset we will build
                rows = n_articles	+	1
                cols = n_words		+	1
                printc(f"attempting construction of matr: ({rows},{cols})","BLUE")
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

                        return matrix.todense(), docwords


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
    matrix, docwords = create_matrix(docNYT,header=3,verbose=True)
    t2 = time()
    printc(f"finished building matrix {matrix.shape} in {t2-t1} seconds", "GREEN")

    ############################################################################
    ################## Calculate the SVD for the matrix ########################
    ############################################################################
    try:
        U, S, Vt = svd(matrix, lapack_driver='gesvd')
    except MemoryError as m:
        printc(f"failed after {time()-t2}","RED")
        printc(m,"RED")
        exit()

    printc(f"finished building SVD in {time()-t2} seconds", "GREEN")
    printc(f"U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}","TAN")

    ############################################################################
    ################### Dimensional Reduction via SVD  #########################
    ############################################################################

    n, k = sparse_matrix.shape


    mean_square_errors = np.zeros((k,1))
    sPrime = np.copy(S)
    Sigma = np.zeros((n,k))

    for i in range(20,1,-1):
        sPrime[i] = 0
        Sigma[0:k,0:k] = np.diag(sPrime)
        print(f"MULT-- U: {U.shape}, S: {Sigma.shape}, Vt: {Vt.shape}")
        Ap1 = U@Sigma@Vt
        mean_square_errors[i] = mean_squared_error(matrix,Ap1,squared=False)

        # plot our data
        print(mean_square_errors)
