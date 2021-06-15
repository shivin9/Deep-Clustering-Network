import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from cac_main import specificity, sensitivity, best_threshold, predict_clusters, predict_clusters_cac,\
compute_euclidean_distance, calculate_gamma_old, calculate_gamma_new,\
cac, get_new_accuracy, score
import umap
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari

def update(X, y, cluster_stats, labels, centers, positive_centers, negative_centers, k, alpha):
    beta = -np.infty
    total_iterations = 100
    errors = np.zeros((total_iterations, k))
    lbls = []
    lbls.append(np.copy(labels))

    for iteration in range(1, total_iterations):
        N = len(X)
        cluster_label = []
        for index_point in range(N):
            distance = {}
            pt = X[index_point]
            pt_label = y[index_point]
            cluster_id = labels[index_point]
            p, n = cluster_stats[cluster_id][0], cluster_stats[cluster_id][1]
            new_cluster = old_cluster = labels[index_point]
            old_err = np.zeros(k)
            # Ensure that degeneracy is not happening
            if ~((p == 1 and pt_label == 1) or (n == 1 and pt_label == 0)):
                for cluster_id in range(0, k):
                    if cluster_id != old_cluster:
                        distance[cluster_id] = calculate_gamma_new(pt, pt_label,\
                                                centers[cluster_id], positive_centers[cluster_id],\
                                                negative_centers[cluster_id], cluster_stats[cluster_id], alpha)
                    else:
                        distance[cluster_id] = np.infty

                old_gamma = calculate_gamma_old(pt, pt_label,\
                                                centers[old_cluster], positive_centers[old_cluster],\
                                                negative_centers[old_cluster], cluster_stats[old_cluster], alpha)
                # new update condition
                new_cluster = min(distance, key=distance.get)
                new_gamma = distance[new_cluster]

                if beta < old_gamma + new_gamma < 0:
                    # Remove point from old cluster
                    p, n = cluster_stats[old_cluster] # Old cluster statistics
                    t = p + n

                    centers[old_cluster] = (t/(t-1))*centers[old_cluster] - (1/(t-1))*pt

                    if pt_label == 0:
                        negative_centers[old_cluster] = (n/(n-1))*negative_centers[old_cluster] - (1/(n-1)) * pt
                        cluster_stats[old_cluster][1] -= 1

                    else:
                        positive_centers[old_cluster] = (p/(p-1))*positive_centers[old_cluster] - (1/(p-1)) * pt
                        cluster_stats[old_cluster][0] -= 1

                    # Add point to new cluster
                    p, n = cluster_stats[new_cluster] # New cluster statistics
                    t = p + n
                    centers[new_cluster] = (t/(t+1))*centers[new_cluster] + (1/(t+1))*pt

                    if pt_label == 0:
                        negative_centers[new_cluster] = (n/(n+1))*negative_centers[new_cluster] + (1/(n+1)) * pt
                        cluster_stats[new_cluster][1] += 1

                    else:
                        positive_centers[new_cluster] = (p/(p+1))*positive_centers[new_cluster] + (1/(p+1)) * pt
                        cluster_stats[new_cluster][0] += 1
                    labels[index_point] = new_cluster

        lbls.append(np.copy(labels))

        for idp in range(N):
            pt = X[idp]
            cluster_id = labels[idp]
            errors[iteration][cluster_id] += compute_euclidean_distance(pt, centers[cluster_id])-alpha*compute_euclidean_distance(positive_centers[cluster_id],\
                                                negative_centers[cluster_id])

        if ((lbls[iteration] == lbls[iteration-1]).all()) and iteration > 0:
            print("converged at itr: ", iteration)
            break
    print("ARI(KM, CAC) = ", ari(lbls[0], lbls[-1]))

    # print(errors, np.sum(errors, axis=1))
    return cluster_stats, labels, centers, positive_centers, negative_centers


class batch_cac(object):    
    def __init__(self, args):
        self.args = args
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters
        self.cluster_stats = np.zeros((self.n_clusters,2))
        self.clusters = np.zeros((self.n_clusters, self.latent_dim))
        self.positive_centers = np.zeros((self.n_clusters, self.latent_dim))
        self.negative_centers = np.zeros((self.n_clusters, self.latent_dim))
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        self.n_jobs = args.n_jobs
        # self.reducer = umap.UMAP()
    
    def cluster(self, X, y, alpha):
        cluster_labels = predict_clusters(X, self.clusters)

        # clusters, models, lbls, errors, seps, loss = \
        # cac(X, cluster_labels, 10, y, alpha, -np.infty, classifier="LR", verbose=False)
        # select the last centers obtained after last clustering
        # clustering = -1
        # update cluster centers after running CAC
        # self.clusters = clusters[1][clustering][0]

        self.cluster_stats, new_labels, self.clusters, self.positive_centers, self.negative_centers = update(X, y, self.cluster_stats, cluster_labels,\
             self.clusters, self.positive_centers, self.negative_centers, self.n_clusters, alpha)

        # new_cluster_labels = predict_clusters(X, self.clusters)
        return new_labels
    
    def init_cluster(self, X, y, indices=None):
        """ Generate initial clusters using sklearn.Kmeans """
        """ X will be AE embeddings """
        model = KMeans(n_clusters=self.n_clusters,
                       n_init=20)
        model.fit(X)
        # X2 = self.reducer.fit_transform(X)
        # color = ['yellow', 'red']
        # c = [color[int(model.labels_[i])] for i in range(len(X2))] 
        # plt.scatter(X2[:,0], X2[:,1], color=c)
        # plt.show()
        self.clusters = model.cluster_centers_  # copy clusters
        labels = model.labels_

        for j in range(self.n_clusters):
            pts_index = np.where(labels == j)[0]
            cluster_pts = X[pts_index]        
            self.clusters[j,:] = cluster_pts.mean(axis=0)
            n_class_index = np.where(y[pts_index] == 0)[0]
            p_class_index = np.where(y[pts_index] == 1)[0]

            self.cluster_stats[j][0] = len(p_class_index)
            self.cluster_stats[j][1] = len(n_class_index)

            n_class = cluster_pts[n_class_index]
            p_class = cluster_pts[p_class_index]

            self.negative_centers[j,:] = n_class.mean(axis=0)
            self.positive_centers[j,:] = p_class.mean(axis=0)

        # self.clusters = np.random.rand(self.n_clusters, self.latent_dim)  # copy clusters

    def update_assign(self, X):
        """ Assign samples in `X` to clusters """
        return predict_clusters(X, self.clusters)