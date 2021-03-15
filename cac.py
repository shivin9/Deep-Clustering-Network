import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from cac_main import specificity, sensitivity, best_threshold, predict_clusters, predict_clusters_cac,\
compute_euclidean_distance, calculate_gamma_old, calculate_gamma_new,\
cac, get_new_accuracy, score
import umap
from matplotlib import pyplot as plt

def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat


class batch_cac(object):
    
    def __init__(self, args):
        self.args = args
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters
        self.clusters = np.zeros((self.n_clusters, self.latent_dim))
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        self.n_jobs = args.n_jobs
        self.reducer = umap.UMAP()
    
    def cluster(self, X, y, alpha):
        # print(len(y), sum(y))
        cluster_labels = predict_clusters(X, self.clusters)
        clusters, models, lbls, errors, seps, loss = \
        cac(X, cluster_labels, 10, y, alpha, -np.infty, classifier="LR", verbose=False)
        # select the last centers obtained after last clustering
        clustering = -1
        # update cluster centers after running CAC
        self.clusters = clusters[1][clustering][0]
        return lbls[clustering]
    
    def init_cluster(self, X, indices=None):
        """ Generate initial clusters using sklearn.Kmeans """
        model = KMeans(n_clusters=self.n_clusters,
                       n_init=20)
        model.fit(X)
        # X2 = self.reducer.fit_transform(X)
        # color = ['yellow', 'red']
        # c = [color[int(model.labels_[i])] for i in range(len(X2))] 
        # plt.scatter(X2[:,0], X2[:,1], color=c)
        # plt.show()
        self.clusters = model.cluster_centers_  # copy clusters
        # self.clusters = np.random.rand(self.n_clusters, self.latent_dim)  # copy clusters

    def update_assign(self, X):
        """ Assign samples in `X` to clusters """
        return predict_clusters(X, self.clusters)