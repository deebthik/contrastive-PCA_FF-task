
#----------| Imports |----------#

import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import os
from PIL import Image
from numpy import linalg as LA
from sklearn import cluster
import sys

import warnings
warnings.filterwarnings("ignore")


#----------Imports----------#


#------------------------| PCA |------------------------#



class PCA():

    def fit_transform(self, X, directions=2):

        # get number of samples and components
        self.n_samples = X.shape[0]
        self.directions = directions

        self.A = X

        # calculate covariance matrix
        covariance_matrix = self.get_covariance_matrix()

        # retrieve selected eigenvectors
        eigenvectors = self.get_eigenvectors(covariance_matrix)

        # project into lower dimension
        projected_matrix = self.project_matrix(eigenvectors)
        return projected_matrix


    def get_covariance_matrix(self, ddof=0):

        # calculate covariance matrix with standardized matrix A
        C = numpy.dot(self.A.T, self.A) / (self.n_samples-ddof)
        return C

    def get_eigenvectors(self, C):
        # calculate eigenvalues & eigenvectors of covariance matrix C
        eigenvalues, eigenvectors = numpy.linalg.eig(C)

        # sort eigenvalues descending and select columns based on directions
        n_cols = numpy.argsort(eigenvalues)[::-1][:self.directions]
        selected_vectors = eigenvectors[:, n_cols]
        return selected_vectors

    def project_matrix(self, eigenvectors):
        P = numpy.dot(self.A, eigenvectors)
        return P



#------------------------PCA------------------------#




A=[]
i = 0

directory = os.fsencode("/Users/deebthik/Desktop/test_FF/final_testdata/final_superimposed_images_6000")

for file in os.listdir(directory):

    #opening Image and resizing to 28x28
    image_test = numpy.array(Image.open("/Users/deebthik/Desktop/test_FF/final_testdata/final_superimposed_images_6000/" + os.fsdecode(file)).resize((28,28)).convert("L"))

    x=numpy.array(image_test)
    #convert to 1D vector
    y=numpy.concatenate(x)

    A += [y]

    i += 1
    print ("Target parsing iteration - " + str(i))
    if i == 5000:
        break

#generate final numpy for target
A_final = numpy.array(A)

#PCA for target
pca = PCA()
converted_data = pca.fit_transform(A_final)

#plotting PCA for target
plt.style.use("dark_background")
plt.figure(figsize = (10,6))

plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 30, alpha = 0.75, c = "cyan")
plt.xlabel("PC-1") , plt.ylabel("PC-2")
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.set_xticks([])
ax.set_yticks([])

#saving PCA for target
plt.savefig("/Users/deebthik/Desktop/bla/PCA.png")





#------------------------| Contrastive PCA |------------------------#




class Contrastive_PCA(object):

    def __init__(self):
        self.directions = 2

    def fit_transform(self, target, background, alpha):

        #initialising class variables
        self.pca_directions = None
        self.target = target
        self.bg = background
        self.n_fg, self.features_d = target.shape
        self.n_bg, self.features_d_bg = background.shape

        #throw error message is background and target dimensions are different
        if (self.features_d != self.features_d_bg):
            print("The dimensions of the target and background datasets must be the same")
            sys.exit(1)

        #getting the avg positioning
        self.bg = self.bg - numpy.mean(self.bg, axis=0)
        self.target = self.target - numpy.mean(self.target, axis=0)

        #calculating the correpsonding covariance matrices
        self.bg_cmatrix = self.bg.T.dot(self.bg)/(self.bg.shape[0]-1)
        self.fg_cmatrix = self.target.T.dot(self.target)/(self.n_fg-1)

        return self.transform(self.target, alpha)


    def transform(self, dataset, alpha):

        self.labels = numpy.ones(dataset.shape[0])

        #calculating the subspace - cPCA
        final_data = []
        directions = self.directions
        formula = self.fg_cmatrix - (alpha * self.bg_cmatrix)
        w, v = LA.eig(formula)
        eig_pointer = numpy.argpartition(w, -directions)[-directions:]
        eig_pointer = eig_pointer[numpy.argsort(-w[eig_pointer])]
        v_top = v[:,eig_pointer]
        final_dataset = dataset.dot(v_top)
        final_dataset[:,0] = final_dataset[:,0] * numpy.sign(final_dataset[0,0])
        final_dataset[:,1] = final_dataset[:,1] * numpy.sign(final_dataset[0,1])
        final_data.append(final_dataset)


        return (final_data, self.labels, alpha)





#------------------------Contrastive PCA------------------------#




B=[]
i = 0
directory = os.fsencode("/Users/deebthik/Desktop/test_FF/final_testdata/satellite_images")

for file in reversed(os.listdir(directory)):

    #ppening Image and resizing to 28x28
    image_test = numpy.array(Image.open("/Users/deebthik/Desktop/test_FF/final_testdata/satellite_images/" + os.fsdecode(file)).resize((28,28)).convert("L"))

    x=numpy.array(image_test)
    #convert to 1D vector
    y=numpy.concatenate(x)

    B += [y]

    i += 1
    print ("Background parsing iteration - " + str(i))

    if i == 5000:
        break

#generating final background numpy
B_final = numpy.array(B)


#plotting cPCA for alpha values 0-1000, in steps of 10 -> 100 plotted graphs
index = 1
for iteration in range (0, 1000, 10):

    cpca = Contrastive_PCA()
    final_data, labels, alpha = cpca.fit_transform(A_final, B_final, iteration)

    for i, j in enumerate(final_data):
            plt.figure(figsize = (10,6))

            for u, w in enumerate(numpy.sort(numpy.unique(labels))):
                pointer = numpy.where(labels==w)
                plt.scatter(j[pointer,0],j[pointer,1], s = 30, alpha = 0.75, c = "cyan")

            plt.title("Alpha="+str(alpha))

            plt.style.use("dark_background")
            ax = plt.gca()
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

            print ("cPCA Iteration - " + str(index))
            print ("Saving cPCA Iteration - " + str(index))
            plt.savefig("/Users/deebthik/Desktop/bla/"+str(index)+".png")

            index += 1
