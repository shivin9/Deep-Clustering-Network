import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
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
        self.positive_sse = np.zeros(self.n_clusters)
        self.negative_sse = np.zeros(self.n_clusters)
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        self.n_jobs = args.n_jobs


    def calculate_gamma_old(self, pt, label, mu, mup, mun, p_sse, n_sse, cluster_stats, beta=1, alpha=2):
        p, n = cluster_stats[0], cluster_stats[1]
        if label == 0:
            mun_new = (n/(n-1))*mun - (1/(n-1))*pt
            mup_new = mup
            new_n_sse = (n/(n-1))*n_sse - (1/(n-1))*np.linalg.norm(pt-mun_new)*np.linalg.norm(pt-mun)
            new_p_sse = p_sse
            n_new = n-1
            p_new = p

        else:
            mup_new = (p/(p-1))*mup - (1/(p-1))*pt
            mun_new = mun
            new_p_sse = (p/(p-1))*p_sse - (1/(p-1))*np.linalg.norm(pt-mup_new)*np.linalg.norm(pt-mup)
            new_n_sse = n_sse
            p_new = p-1
            n_new = n

        mu_new = (p_new*mup_new + n_new*mun_new)/(p_new + n_new)
        new_lin_sep = np.sum(np.square(mun_new - mup_new))/(new_n_sse + new_p_sse)
        lin_sep = np.sum(np.square(mun - mup))/(n_sse + p_sse)
        mu_sep = np.sum(np.square(mu - mu_new))
        gamma_p = -beta*np.sum(np.square(mu-pt)) - (p+n-1) * mu_sep + (p+n) * alpha*lin_sep - (p+n-1)*alpha*new_lin_sep
        # gamma_p = -np.sum(np.square(mu-pt)) - (p+n-1) * mu_sep + alpha*lin_sep - alpha*new_lin_sep
        return gamma_p


    def calculate_gamma_new(self, pt, label, mu, mup, mun, p_sse, n_sse, cluster_stats, beta=1, alpha=2):
        p, n = cluster_stats[0], cluster_stats[1]
        if label == 0:
            mun_new = (n/(n+1))*mun + (1/(n+1))*pt
            mup_new = mup
            new_n_sse = (n/(n+1))*n_sse + (1/(n+1))*np.linalg.norm(pt-mun_new)*np.linalg.norm(pt-mun)
            new_p_sse = p_sse
            n_new = n+1
            p_new = p

        else:
            mup_new = (p/(p+1))*mup + (1/(p+1))*pt
            mun_new = mun
            new_p_sse = (p/(p+1))*p_sse + (1/(p+1))*np.linalg.norm(pt-mup_new)*np.linalg.norm(pt-mup)
            new_n_sse = n_sse
            p_new = p+1
            n_new = n

        mu_new = (p_new*mup_new + n_new*mun_new)/(p_new + n_new)
        new_lin_sep = np.sum(np.square(mun_new - mup_new))/(new_n_sse + new_p_sse)
        lin_sep = np.sum(np.square(mun - mup))/(n_sse + p_sse)
        mu_sep = np.sum(np.square(mu - mu_new))

        gamma_j = beta*np.sum(np.square(mu_new-pt)) + (p+n)*mu_sep + (p+n) * alpha*lin_sep - (p+n+1)*alpha*new_lin_sep
        # gamma_j = np.sum(np.square(mu_new-pt)) + (p+n)*mu_sep + alpha*lin_sep - alpha*new_lin_sep
        return gamma_j

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
            self.cluster_stats[j][0] = len(p_class_index)
            self.cluster_stats[j][1] = len(n_class_index)
            n_class = cluster_pts[n_class_index]
            p_class = cluster_pts[p_class_index]
            self.negative_sse[j] = np.square(np.linalg.norm(n_class - self.negative_centers[j]))
            self.positive_sse[j] = np.square(np.linalg.norm(p_class - self.positive_centers[j]))

        return None


    def update(self, X, y, labels, beta, alpha):
        total_iterations = 100
        k = self.n_clusters
        errors = np.zeros((total_iterations, k))
        lbls = []
        lbls.append(np.copy(labels))

        if len(np.unique(y)) == 1:
            return cluster_stats, labels, self.clusters, self.positive_centers, self.negative_centers

        old_p, old_n = np.copy(self.positive_centers), np.copy(self.negative_centers)

        for iteration in range(0, total_iterations):
            # print(cluster_stats)
            N = X.shape[0]
            cluster_label = []
            for index_point in range(N):
                distance = {}
                pt = X[index_point]
                pt_label = y[index_point]
                cluster_id = labels[index_point]
                p, n = self.cluster_stats[cluster_id][0], self.cluster_stats[cluster_id][1]
                new_cluster = old_cluster = labels[index_point]
                old_err = np.zeros(k)
                # Ensure that degeneracy is not happening
                if ((p > 2 and pt_label == 1) or (n > 2 and pt_label == 0)):
                    for cluster_id in range(0, k):
                        if cluster_id != old_cluster:
                            distance[cluster_id] = self.calculate_gamma_new(pt, pt_label, self.clusters[cluster_id], self.positive_centers[cluster_id],\
                                                    self.negative_centers[cluster_id], self.positive_sse[cluster_id], self.negative_sse[cluster_id], self.cluster_stats[cluster_id], beta, alpha)
                        else:
                            distance[cluster_id] = np.infty

                    old_gamma = self.calculate_gamma_old(pt, pt_label, self.clusters[old_cluster], self.positive_centers[old_cluster],\
                                                    self.negative_centers[old_cluster], self.positive_sse[old_cluster], self.negative_sse[old_cluster], self.cluster_stats[old_cluster], beta, alpha)
                    # new update condition
                    new_cluster = min(distance, key=distance.get)
                    new_gamma = distance[new_cluster]

                    if old_gamma + new_gamma < 0:
                        # Remove point from old cluster
                        p, n = self.cluster_stats[old_cluster] # Old cluster statistics
                        t = p + n

                        self.clusters[old_cluster] = (t/(t-1))*self.clusters[old_cluster] - (1/(t-1))*pt

                        if pt_label == 0:
                            new_mean = (n/(n-1))*self.negative_centers[old_cluster] - (1/(n-1)) * pt
                            old_mean = self.negative_centers[old_cluster]
                            self.negative_sse[old_cluster] = (n/(n-1))*self.negative_sse[old_cluster] - \
                                    (1/(n-1))*np.linalg.norm(pt-new_mean)*np.linalg.norm(pt-old_mean)
                            self.negative_centers[old_cluster] = new_mean
                            self.cluster_stats[old_cluster][1] -= 1

                        else:
                            new_mean = (p/(p-1))*self.positive_centers[old_cluster] - (1/(p-1)) * pt
                            old_mean = self.positive_centers[old_cluster]
                            self.positive_sse[old_cluster] = (p/(p-1))*self.positive_sse[old_cluster] - \
                                    (1/(p-1)) * np.linalg.norm(pt-new_mean)*np.linalg.norm(pt-old_mean)
                            self.positive_centers[old_cluster] = new_mean
                            self.cluster_stats[old_cluster][0] -= 1


                        # Add point to new cluster
                        p, n = self.cluster_stats[new_cluster] # New cluster statistics
                        t = p + n
                        self.clusters[new_cluster] = (t/(t+1))*self.clusters[new_cluster] + (1/(t+1))*pt

                        if pt_label == 0:
                            new_mean = (n/(n+1))*self.negative_centers[new_cluster] + (1/(n+1)) * pt
                            old_mean = self.negative_centers[new_cluster]
                            self.negative_sse[new_cluster] = (n/(n+1))*self.negative_sse[new_cluster] + \
                                    (1/(n+1)) * np.linalg.norm(pt-new_mean) * np.linalg.norm(pt-old_mean)
                            self.negative_centers[new_cluster] = new_mean
                            self.cluster_stats[new_cluster][1] += 1

                        else:
                            new_mean = (p/(p+1))*self.positive_centers[new_cluster] + (1/(p+1)) * pt
                            old_mean = self.positive_centers[new_cluster]
                            self.positive_sse[new_cluster] = (p/(p+1))*self.positive_sse[new_cluster] + \
                                    (1/(p+1)) * np.linalg.norm(pt-new_mean) * np.linalg.norm(pt-old_mean)
                            self.positive_centers[new_cluster] = new_mean
                            self.cluster_stats[new_cluster][0] += 1

                        labels[index_point] = new_cluster


            lbls.append(np.copy(labels))

            if ((lbls[iteration] == lbls[iteration-1]).all()) and iteration > 0:
                print("converged at itr: ", iteration)
                break

#         print("Mean of Delta Clusters")
#         print(np.mean(old_p-positive_centers), np.mean(old_n - negative_centers))

        return labels

    
    def cluster(self, X, y, beta, alpha):
        # Update assigned cluster labels to points
        cluster_labels = self.predict_clusters(X, self.clusters)

        # Do we need this really? ... yes
        self.update_cluster_centers(X, y, cluster_labels)

        # update cluster centers
        new_labels = self.update(X, y, cluster_labels, beta, alpha)

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

            self.negative_sse[j] = np.square(np.linalg.norm(n_class - self.negative_centers[j]))
            self.positive_sse[j] = np.square(np.linalg.norm(p_class - self.positive_centers[j]))

        # self.clusters = np.random.rand(self.n_clusters, self.latent_dim)  # copy clusters

    def update_assign(self, X, target=None):
        """ Assign samples in `X` to clusters """
        return self.predict_clusters(X, self.clusters)