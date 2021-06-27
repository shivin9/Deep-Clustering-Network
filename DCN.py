import torch
import numbers
import numpy as np
import torch.nn as nn
from cac import batch_cac
from kmeans import batch_KMeans
from meanshift import batch_MeanShift
from autoencoder import AutoEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import davies_bouldin_score as dbs, adjusted_rand_score as ari
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score, roc_auc_score, roc_curve, matthews_corrcoef as mcc
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, RidgeClassifier
from matplotlib import pyplot as plt

color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']

class DCN(nn.Module):
    def __init__(self, args):
        super(DCN, self).__init__()
        self.args = args
        self.beta = args.beta  # coefficient of the clustering term 
        self.lamda = args.lamda  # coefficient of the reconstruction term
        self.device = torch.device(args.device)
        
        # Validation check
        if not self.beta > 0:
            msg = 'beta should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.beta))
        
        if not self.lamda > 0:
            msg = 'lamda should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.lamda))
        
        if len(self.args.hidden_dims) == 0:
            raise ValueError('No hidden layer specified.')
        
        if args.clustering == 'kmeans':
            self.clustering = batch_KMeans(args)
        elif args.clustering == 'meanshift':
            self.clustering = batch_MeanShift(args)
        elif args.clustering == "cac":
            self.clustering = batch_cac(args)
            self.classifier = args.classifier
            self.cluster_classifiers = []
            self.base_classifier = []
        else:
            raise RuntimeError('Error: no clustering chosen')
            
        self.autoencoder = AutoEncoder(args).to(self.device)
        self.criterion  = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.wd)
    
    def get_classifier(self, classifier):
        if classifier == "LR":
            model = LogisticRegression(random_state=0, max_iter=1000)
        elif classifier == "RF":
            model = RandomForestClassifier(n_estimators=10)
        elif classifier == "SVM":
            # model = SVC(kernel="linear", probability=True)
            model = LinearSVC(max_iter=5000)
            model.predict_proba = lambda X: np.array([model.decision_function(X), model.decision_function(X)]).transpose()
        elif classifier == "Perceptron":
            model = Perceptron()
            model.predict_proba = lambda X: np.array([model.decision_function(X), model.decision_function(X)]).transpose()
        elif classifier == "ADB":
            model = AdaBoostClassifier(n_estimators = 100)
        elif classifier == "DT":
            model = DecisionTreeClassifier()
        elif classifier == "LDA":
            model = LDA()
        elif classifier == "NB":
            model = MultinomialNB()
        elif classifier == "SGD":
            model = SGDClassifier(loss='log')
        elif classifier == "Ridge":
            model = RidgeClassifier()
            model.predict_proba = lambda X: np.array([model.decision_function(X), model.decision_function(X)]).transpose()
        elif classifier == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            model = LogisticRegression(class_weight='balanced', max_iter=1000)
        return model

    """ Compute the Equation (5) in the original paper on a data batch """
    def _loss(self, X, cluster_id):
        batch_size = X.size()[0]
        rec_X = self.autoencoder(X)
        latent_X = self.autoencoder(X, latent=True)
        
        # Reconstruction error
        rec_loss = self.lamda * self.criterion(X, rec_X)
        
        # Regularization term on clustering
        km_loss = torch.tensor(0.).to(self.device)
        sep_loss = torch.tensor(0.).to(self.device)
        clusters = torch.FloatTensor(self.clustering.clusters).to(self.device)
        positive_clusters = torch.FloatTensor(self.clustering.positive_centers).to(self.device)
        negative_clusters = torch.FloatTensor(self.clustering.negative_centers).to(self.device)

        for i in range(batch_size):
            diff_vec = latent_X[i] - clusters[cluster_id[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                            diff_vec.view(-1, 1))
            km_loss += 0.5 * self.beta * torch.squeeze(sample_dist_loss)
            
            if self.args.clustering == "cac":
                diff_vec = positive_clusters[cluster_id[i]] - negative_clusters[cluster_id[i]]
                sample_sep_loss = torch.matmul(diff_vec.view(1, -1),
                                                diff_vec.view(-1, 1))
                sep_loss -= np.log(self.args.alpha * torch.squeeze(sample_sep_loss))
        
        for j in range(self.args.n_clusters):
            diff_vec = positive_clusters[j] - negative_clusters[j]
            sample_sep_loss = torch.matmul(diff_vec.view(1, -1),
                                            diff_vec.view(-1, 1))
            sample_sep_loss = torch.squeeze(sample_sep_loss)
            print("Class Dist: ", np.log(sample_sep_loss))

        print("KM Dist: ", km_loss)
        print("Sep Dist: ", sep_loss)

        return (rec_loss + km_loss + sep_loss,
                rec_loss.detach().cpu().numpy(),
                (km_loss + sep_loss).detach().cpu().numpy())
    
    def pretrain(self, train_loader, epoch=100, verbose=True):
        if not self.args.pretrain:
            return
        
        if not isinstance(epoch, numbers.Integral):
            msg = '`epoch` should be an integer but got value = {}'
            raise ValueError(msg.format(epoch))
        
        if verbose:
            print('========== Start pretraining ==========')
        
        rec_loss_list = []
        
        self.train()
        for e in range(epoch):
            for batch_idx, (data, _) in enumerate(train_loader):
                batch_size = data.size()[0]
                data = data.to(self.device).view(batch_size, -1)
                rec_X = self.autoencoder(data)
                loss = self.criterion(data, rec_X)
                if verbose and (batch_idx+1) % self.args.log_interval == 0:
                    msg = 'Epoch: {:02d} | Batch: {:03d} | Rec-Loss: {:.3f}'
                    print(msg.format(e, batch_idx+1, 
                                     loss.detach().cpu().numpy()))
                    rec_loss_list.append(loss.detach().cpu().numpy())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.eval()
        
        if verbose:
            print('========== End pretraining ==========\n')

        self.pre_cluster(train_loader)
                
        return rec_loss_list


    def pre_cluster(self, train_loader):
        # Initialize clusters in self.clustering after pre-training
        batch_X = []
        batch_y = []
        for batch_idx, (data, y) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.to(self.device).view(batch_size, -1)
            latent_X = self.autoencoder(data, latent=True)
            batch_X.append(latent_X.detach().cpu().numpy())
            batch_y.extend(y.detach().cpu().numpy())

        batch_X = np.vstack(batch_X)
        batch_y = np.array(batch_y)

        self.clustering.init_cluster(batch_X, batch_y)
        return None


    def fit(self, epoch, train_loader, verbose=True):
        X_train = []
        y_train = []
        cluster_ids_train = []
        for batch_idx, (data, y) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.view(batch_size, -1).to(self.device)

            # Collect training data and labels for the later classifier
            X_train.append(data.cpu().numpy())
            y_train.extend(y.numpy())
            
            # Get the latent features
            with torch.no_grad():
                latent_X = self.autoencoder(data, latent=True)
                latent_X = latent_X.cpu().numpy()

            if self.args.clustering == "cac":
                cluster_id = self.clustering.cluster(latent_X, y, self.args.beta, self.args.alpha)

            else:
                # [Step-1] Update the assignment results
                cluster_id = self.clustering.update_assign(latent_X, y)

                # [Step-2] Update cluster centers in batch Clustering
                elem_count = np.bincount(cluster_id,
                                         minlength=self.args.n_clusters)

                for k in range(self.args.n_clusters):
                    # avoid empty slicing
                    if elem_count[k] == 0:
                        continue
                    # updating the cluster center
                    self.clustering.update_cluster(latent_X[cluster_id == k], k)
            
            # [Step-3] Update the network parameters
            loss, rec_loss, dist_loss = self._loss(data, cluster_id)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

#             if verbose and (batch_idx+1) % self.args.log_interval == 0:
            msg = 'Epoch: {:02d} | Batch: {:03d} | Loss: {:.3f} | Rec-' \
                  'Loss: {:.3f} | Dist-Loss: {:.3f}'
            print(msg.format(epoch, batch_idx+1, 
                             loss.detach().cpu().numpy(),
                             rec_loss, dist_loss))

        X_train = np.vstack(X_train)
        self.eval()
        if self.args.clustering == "cac":
            with torch.no_grad():
                latent_X_train = self.autoencoder(torch.FloatTensor(np.array(X_train)).to(self.args.device), latent=True)
                latent_X_train = latent_X_train.to(self.args.device).numpy()

            cluster_ids_train = self.clustering.update_assign(latent_X_train)
            y_train = np.array(y_train)
            X_train = latent_X_train
            
            print("Training Base classifier")
            classifier = self.get_classifier(self.classifier)
            classifier.fit(X_train, y_train)
            self.base_classifier.append(classifier)
            print("Base Training F1:", f1_score(y_train, classifier.predict(X_train).ravel()))
            print("Base Training MCC:", mcc(y_train, classifier.predict(X_train).ravel()))
            print("Base Training AUC:", roc_auc_score(y_train, classifier.predict_proba(X_train)[:,1]))

            print("Training CAC classifiers")

            self.cluster_classifiers.append([])
            y_pred = []
            y_true = []
            y_pred_proba = []
            for j in range(self.args.n_clusters):
                cluster_indices = np.where(cluster_ids_train == j)[0]
                X_cluster = X_train[cluster_indices]
                y_cluster = y_train[cluster_indices]
                y_true.extend(y_cluster)
                classifier = self.get_classifier(self.classifier)
                if np.unique(y_cluster).shape[0] > 1:
                    classifier.fit(X_cluster, y_cluster.ravel())
                    print("CAC Training F1:", f1_score(y_cluster, classifier.predict(X_cluster)))
                    print("CAC Training MCC:", mcc(y_cluster, classifier.predict(X_cluster)))
                    print("CAC Training AUC:", roc_auc_score(y_cluster, classifier.predict_proba(X_cluster)[:,1]))
                    y_pred.extend(classifier.predict(X_cluster))
                    y_pred_proba.extend(classifier.predict_proba(X_cluster)[:,1])
                else:
                    print("Fitting random classifier, Iteration:", j)
                    tmp = np.random.randint(2,size=y_cluster.shape[0])
                    y_pred.extend(tmp)
                    y_pred_proba.extend(tmp)
                    classifier.fit(X_cluster, tmp)
                self.cluster_classifiers[-1].append(classifier)
            print("Final CAC Training F1:", f1_score(y_true, y_pred))
            print("Final CAC Training MCC:", mcc(y_true, y_pred))
            print("Final CAC Training AUC:", roc_auc_score(y_true, y_pred_proba))