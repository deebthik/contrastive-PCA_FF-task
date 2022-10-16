
#----------Imports----------#

import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import os
from PIL import Image
from numpy import linalg as LA
from sklearn import cluster
import sys


#----------Imports----------#


#------------------------PCA------------------------#

class PCA():

    def fit_transform(self, X, n_components=2):
        # get number of samples and components
        self.n_samples = X.shape[0]
        self.n_components = n_components

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
        # calculate eigenvalues & eigenvectors of covariance matrix 'C'
        eigenvalues, eigenvectors = numpy.linalg.eig(C)

        # sort eigenvalues descending and select columns based on n_components
        n_cols = numpy.argsort(eigenvalues)[::-1][:self.n_components]
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
    #Opening Image and resizing to 10X10 for easy viewing
    image_test = numpy.array(Image.open('/Users/deebthik/Desktop/test_FF/final_testdata/final_superimposed_images_6000/' + os.fsdecode(file)).resize((28,28)).convert('L'))  #note: I used a local image
    #print image
    #print (image_test)

    #manipulate the array
    x=numpy.array(image_test)
    #convert to 1D vector
    y=numpy.concatenate(x)
    #print (len(y))
    A += [y]

    i += 1
    print ("A iteration - " + str(i))
    if i == 100:
        break


A_final = numpy.array(A)


pca = PCA() # we need 2 principal components.
converted_data = pca.fit_transform(A_final)


plt.style.use('dark_background')
plt.figure(figsize = (10,6))
#c_map = plt.cm.get_cmap('jet', 10)
plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 15, c = 'cyan')
#plt.colorbar()
plt.xlabel('PC-1') , plt.ylabel('PC-2')
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.set_xticks([])
ax.set_yticks([])
#plt.show()
plt.savefig('/Users/deebthik/Desktop/pca.png')




#------------------------Contrastive PCA------------------------#




class Contrastive_PCA(object):

    def __init__(self, n_components=2):
        self.n_components = n_components
        #self.fitted = False

    def fit_transform(self, target, background, alpha_selection='auto', n_alphas=40, max_log_alpha=3, n_alphas_to_return=4, active_labels = None, legend=None, alpha_value=None, return_alphas=False):

        self.pca_directions = None
        self.bg_eig_vals = None
        self.affinity_matrix = None

        self.fg = target
        self.bg = background
        self.n_fg, self.features_d = target.shape
        self.n_bg, self.features_d_bg = background.shape

        if (self.features_d != self.features_d_bg):
            print("The dimensions of the target and background datasets must be the same")
            sys.exit(1)


        #Center the background and target data
        self.bg = self.bg - numpy.mean(self.bg, axis=0)
        self.fg = self.fg - numpy.mean(self.fg, axis=0)


        #Calculate the covariance matrices
        self.bg_cov = self.bg.T.dot(self.bg)/(self.bg.shape[0]-1)
        self.fg_cov = self.fg.T.dot(self.fg)/(self.n_fg-1)


        #self.fitted = True
        
        
        return self.transform(dataset=self.fg, alpha_selection=alpha_selection, n_alphas=n_alphas, max_log_alpha=max_log_alpha, n_alphas_to_return=n_alphas_to_return, active_labels=active_labels, legend=legend, alpha_value=alpha_value, return_alphas=return_alphas)
        


    def transform(self, dataset, alpha_selection='auto', n_alphas=40, max_log_alpha=3, n_alphas_to_return=4, active_labels = None, legend=None, alpha_value=None, return_alphas=False):
        
        if active_labels is None:
            active_labels = numpy.ones(dataset.shape[0])
        self.active_labels = active_labels


        if (alpha_selection=='auto'):
            transformed_data, best_alphas = self.automated_cpca(dataset, n_alphas_to_return, n_alphas, max_log_alpha)

                            
            for j, fg in enumerate(transformed_data):
                plt.figure(figsize = (10,6))
                for i, l in enumerate(numpy.sort(numpy.unique(self.active_labels))):
                    idx = numpy.where(self.active_labels==l)
                    plt.scatter(fg[idx,0],fg[idx,1], alpha=1, c = 'cyan')
                plt.title('Alpha='+str(numpy.round(best_alphas[j],2)))

                plt.style.use('dark_background')
                ax = plt.gca()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
            
                plt.show()

        return



    """
    This function performs contrastive PCA using the alpha technique on the
    active and background dataset. It automatically determines n_alphas=4 important values
    of alpha up to based to the power of 10^(max_log_alpha=5) on spectral clustering
    of the top subspaces identified by cPCA.
    The final return value is the data projected into the top (n_components = 2)
    subspaces, which can be plotted outside of this function
    """
    def automated_cpca(self, dataset, n_alphas_to_return, n_alphas, max_log_alpha):
        best_alphas, all_alphas, _, _ = self.find_spectral_alphas(n_alphas, max_log_alpha, n_alphas_to_return)
        best_alphas = numpy.concatenate(([0], best_alphas)) #one of the alphas is always alpha=0
        data_to_plot = []
        for alpha in best_alphas:
            transformed_dataset = self.cpca_alpha(dataset=dataset, alpha=alpha)
            data_to_plot.append(transformed_dataset)
        return data_to_plot, best_alphas

    """
    This function performs contrastive PCA using the alpha technique on the
    active and background dataset. It returns the cPCA-reduced data for all values of alpha specified,
    both the active and background, as well as the list of alphas
    """
    def all_cpca(self, dataset, n_alphas, max_log_alpha):
        alphas = numpy.concatenate(([0],numpy.logspace(-1,max_log_alpha,n_alphas)))
        data_to_plot = []
        for alpha in alphas:
            transformed_dataset = self.cpca_alpha(dataset=dataset, alpha=alpha)
            data_to_plot.append(transformed_dataset)
        return data_to_plot, alphas

    """
    Returns active and bg dataset projected in the cpca direction, as well as the top c_cpca eigenvalues indices.
    If specified, it returns the top_cpca directions
    """
    def cpca_alpha(self, dataset, alpha=1):
        n_components = self.n_components
        sigma = self.fg_cov - alpha*self.bg_cov
        w, v = LA.eig(sigma)
        eig_idx = numpy.argpartition(w, -n_components)[-n_components:]
        eig_idx = eig_idx[numpy.argsort(-w[eig_idx])]
        v_top = v[:,eig_idx]
        reduced_dataset = dataset.dot(v_top)
        reduced_dataset[:,0] = reduced_dataset[:,0]*numpy.sign(reduced_dataset[0,0])
        reduced_dataset[:,1] = reduced_dataset[:,1]*numpy.sign(reduced_dataset[0,1])
        return reduced_dataset

    """
    This method performs spectral clustering on the affinity matrix of subspaces
    returned by contrastive pca, and returns (`=3) exemplar values of alpha
    """
    def find_spectral_alphas(self, n_alphas, max_log_alpha, n_alphas_to_return):
        self.create_affinity_matrix(max_log_alpha, n_alphas)
        affinity = self.affinity_matrix
        spectral = cluster.SpectralClustering(n_clusters=n_alphas_to_return, affinity='precomputed')
        alphas = numpy.concatenate(([0],numpy.logspace(-1,max_log_alpha,n_alphas)))
        spectral.fit(affinity)
        labels = spectral.labels_
        best_alphas = list()
        for i in range(n_alphas_to_return):
            idx = numpy.where(labels==i)[0]
            if not(0 in idx): #because we don't want to include the cluster that includes alpha=0
                affinity_submatrix = affinity[idx][:, idx]
                sum_affinities = numpy.sum(affinity_submatrix, axis=0)
                exemplar_idx = idx[numpy.argmax(sum_affinities)]
                best_alphas.append(alphas[exemplar_idx])
        return numpy.sort(best_alphas), alphas, affinity[0,:], labels

    """
    This method creates the affinity matrix of subspaces returned by contrastive pca
    """
    def create_affinity_matrix(self, max_log_alpha, n_alphas):
        from math import pi
        alphas = numpy.concatenate(([0],numpy.logspace(-1,max_log_alpha,n_alphas)))
        subspaces = list()
        k = len(alphas)
        affinity = 0.5*numpy.identity(k) #it gets doubled
        for alpha in alphas:
            space = self.cpca_alpha(dataset=self.fg, alpha=alpha)
            q, r = numpy.linalg.qr(space)
            subspaces.append(q)
        for i in range(k):
            for j in range(i+1,k):
                q0 = subspaces[i]
                q1 = subspaces[j]
                u, s, v = numpy.linalg.svd(q0.T.dot(q1))
                affinity[i,j] = s[0]*s[1]
        affinity = affinity + affinity.T
        self.affinity_matrix = numpy.nan_to_num(affinity)





#------------------------Contrastive PCA------------------------#




B=[]
i = 0
directory = os.fsencode("/Users/deebthik/Desktop/test_FF/final_testdata/satellite_images")

for file in reversed(os.listdir(directory)):
    
    #Opening Image and resizing to 10X10 for easy viewing
    image_test = numpy.array(Image.open('/Users/deebthik/Desktop/test_FF/final_testdata/satellite_images/' + os.fsdecode(file)).resize((28,28)).convert('L'))  #note: I used a local image
    #print image
    #print (image_test)

    #manipulate the array
    x=numpy.array(image_test)
    #convert to 1D vector
    y=numpy.concatenate(x)
    #print (len(y))
    B += [y]
    
    i += 1
    print ("B iteration - " + str(i))

    if i == 100:
        break
    
    

B_final = numpy.array(B)


#A_labels = [0]*300+[1]*300+[2]*300
#A_labels = [0]*200+[1]*200+[2]*200+[3]*200+[4]*200


cpca = Contrastive_PCA()
cpca.fit_transform(A_final, B_final)
                                
