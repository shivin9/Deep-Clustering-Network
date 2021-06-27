import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from cac_main import specificity, sensitivity, best_threshold, cac,\
compute_euclidean_distance, calculate_gamma_old, calculate_gamma_new
import umap
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari

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


    def predict_clusters(self, X_test, centers) -> np.array:
        K = centers.shape[0]
        dists = np.zeros(K)
        test_labels = np.zeros(X_test.shape[0])

        for pt in range(X_test.shape[0]):
            for k in range(K):
                min_dist = np.square(np.linalg.norm(centers[k] - X_test[pt]))
                dists[k] = min_dist
            test_labels[pt] = int(np.argmin(dists))
        return test_labels.astype(int)


    def update_cluster_centers(self, X, y, cluster_labels):
        for j in range(self.n_clusters):
            pts_index = np.where(cluster_labels == j)[0]
            cluster_pts = X[pts_index]        
            for pt in pts_index:
                self.count[j] += 1
                eta = 1/(self.count[j])
                self.clusters[j,:] = (1-eta)*self.clusters[j,:] + eta*X[pt]

                if y[pt] == 0:
                    self.negative_centers[j,:] = (1-eta)*self.negative_centers[j,:] +\
                                                    eta*X[pt]
                else:
                    self.positive_centers[j,:] = (1-eta)*self.positive_centers[j,:] +\
                                                    eta*X[pt]

            n_class_index = np.where(y[pts_index] == 0)[0]
            p_class_index = np.where(y[pts_index] == 1)[0]

#             self.cluster_stats[j][0] = len(p_class_index)
#             self.cluster_stats[j][1] = len(n_class_index)

        return None


    def update(self, X, y, cluster_stats, labels, centers, positive_centers, negative_centers, k, beta, alpha):
        total_iterations = 100
        errors = np.zeros((total_iterations, k))
        lbls = []
        lbls.append(np.copy(labels))

        if len(np.unique(y)) == 1:
            return cluster_stats, labels, centers, positive_centers, negative_centers

        for iteration in range(total_iterations):
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
                if ((p > 1 and pt_label == 1) or (n > 1 and pt_label == 0)):
                    for cluster_id in range(0, k):
                        if cluster_id != old_cluster:
                            distance[cluster_id] = calculate_gamma_new(pt, pt_label, centers[cluster_id],\
                                                    positive_centers[cluster_id], negative_centers[cluster_id],\
                                                    cluster_stats[cluster_id], beta, alpha)
                        else:
                            distance[cluster_id] = np.infty

                    old_gamma = calculate_gamma_old(pt, pt_label, centers[old_cluster],\
                                                    positive_centers[old_cluster], negative_centers[old_cluster],\
                                                    cluster_stats[old_cluster], beta, alpha)
                    # new update condition
                    new_cluster = min(distance, key=distance.get)
                    new_gamma = distance[new_cluster]

                    if old_gamma + new_gamma < 0:
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
                        assert(cluster_stats[old_cluster][0] > 0)
                        assert(cluster_stats[old_cluster][1] > 0)

            lbls.append(np.copy(labels))

            if ((lbls[iteration] == lbls[iteration-1]).all()) and iteration > 0:
#                 print("converged at itr: ", iteration)
                break

        return cluster_stats, labels, centers, positive_centers, negative_centers

    
    def cluster(self, X, y, beta, alpha):
        # Update assigned cluster labels to points
        cluster_labels = self.predict_clusters(X, self.clusters)

        # Do we need this really? ... yes
        self.update_cluster_centers(X, y, cluster_labels)

        # update cluster centers
        self.cluster_stats, new_labels, self.clusters, self.positive_centers, self.negative_centers = self.update(X, y, self.cluster_stats, cluster_labels,\
             self.clusters, self.positive_centers, self.negative_centers, self.n_clusters, beta, alpha)

        return new_labels
    

    def init_cluster(self, X, y, indices=None):
        """ Generate initial clusters using sklearn.Kmeans """
        """ X will be AE embeddings """
        model = KMeans(n_clusters=self.n_clusters,
                       n_init=20)
        model.fit(X)
        self.clusters = model.cluster_centers_  # copy clusters
        labels = model.labels_

        for j in range(self.n_clusters):
            pts_index = np.where(labels == j)[0]
            cluster_pts = X[pts_index]
#             assert(np.allclose(self.clusters[j,:], cluster_pts.mean(axis=0)))
            n_class_index = np.where(y[pts_index] == 0)[0]
            p_class_index = np.where(y[pts_index] == 1)[0]

            self.cluster_stats[j][0] = len(p_class_index)
            self.cluster_stats[j][1] = len(n_class_index)

            n_class = cluster_pts[n_class_index]
            p_class = cluster_pts[p_class_index]

            self.negative_centers[j,:] = n_class.mean(axis=0)
            self.positive_centers[j,:] = p_class.mean(axis=0)

        # self.clusters = np.random.rand(self.n_clusters, self.latent_dim)  # copy clusters

    def update_assign(self, X, target=None):
        """ Assign samples in `X` to clusters """
        return self.predict_clusters(X, self.clusters)