import torch
import argparse
import numpy as np
from DCN import DCN
from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score, roc_auc_score
from torch.utils.data import Subset
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pacmap
from sklearn.datasets import make_blobs

color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']

def evaluate(model, test_loader):
    X_test = np.empty(shape=model.args.latent_dim)
    y_test = []
    y_pred = []
    y_classifier_pred = []
    y_classifier_pred_proba = []

    for data, target in test_loader:
        batch_size = data.size()[0]
        data = data.view(batch_size, -1).to(model.device)
        latent_X = model.autoencoder(data, latent=True)
        latent_X = latent_X.detach().cpu().numpy()
        X_test = np.vstack([X_test, latent_X])
        y_test.append(target.view(-1, 1).numpy())
        y_pred.append(model.clustering.update_assign(latent_X, target).reshape(-1, 1))
    
    y_test = np.vstack(y_test).reshape(-1)
    y_pred = np.vstack(y_pred).reshape(-1)
    nmi, ari = normalized_mutual_info_score(y_test, y_pred), adjusted_rand_score(y_test, y_pred)

    base_f1 = f1_score(y_test, model.base_classifier[-1].predict(X_test))
    base_auc = roc_auc_score(y_test, model.base_classifier[-1].predict_proba(X_test)[:,1])
    
    X_cluster_test = []
    y_cluster_test = []

    for j in range(model.args.n_clusters):
        cluster_index = np.where(y_pred == j)[0]
        X_cluster = X_test[cluster_index]
        y_cluster = y_test[cluster_index]

        X_cluster_test.append(X_cluster)
        y_cluster_test.append(y_cluster)

        # Select the cluster classifiers appearing in the latest iteration
        y_classifier_pred.append(model.cluster_classifiers[-1][j].predict(X_cluster))
        y_classifier_pred_proba.append(model.cluster_classifiers[-1][j].predict_proba(X_cluster)[:,1])

    cac_f1 = f1_score(y_cluster_test, y_classifier_pred)
    cac_auc = roc_auc_score(y_test, y_classifier_pred_proba)

    return (nmi, ari, base_f1, base_auc, cac_f1, cac_auc)

def solver(args, model, train_loader, test_loader):
    rec_loss_list = model.pretrain(train_loader, epoch=args.pre_epoch)
    nmi_list = []
    ari_list = []

    for e in range(args.epoch):
        # Show training set
        if e%1 == 0:
            out = model.autoencoder(torch.FloatTensor(np.array(X_train)).to(args.device), latent=True)
            cluster_id = model.clustering.update_assign(out.cpu().detach().numpy())
            X2 = reducer.fit_transform(out.cpu().detach().numpy())

    #         X_centers = reducer.transform(model.clustering.clusters)

            c_clusters = [color[int(cluster_id[i])] for i in range(len(cluster_id))]
            c_labels = [color[int(y_train[i])] for i in range(len(cluster_id))]
            # plt.scatter(latent_X[:,0], latent_X[:,1], color=c_train); plt.show()

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Clusters vs Labels')
            ax1.scatter(X2[:,0], X2[:,1], color=c_clusters)
    #         ax1.plot(X_centers[0], marker='x', markersize=3, color="green")
    #         ax1.plot(X_centers[1], marker='x', markersize=3, color="green")

            ax2.scatter(X2[:,0], X2[:,1], color=c_labels)
            plt.show()

            # Print testset
            out = model.autoencoder(torch.FloatTensor(np.array(X_test)).to(args.device), latent=True)
            test_cluster_id = model.clustering.update_assign(out.cpu().detach().numpy())
            X_t = reducer.transform(out.cpu().detach().numpy())

            c_clusters = [color[int(test_cluster_id[i])] for i in range(len(test_cluster_id))]
            c_test = [color[int(y_test[i])] for i in range(len(y_test))]

            figure = plt.figure()
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('CAC Testing Clusters vs Testing Embeddings')
            ax1.scatter(X_t[:,0], X_t[:,1], color=c_clusters)
            ax2.scatter(X_t[:,0], X_t[:,1], color=c_test)
            plt.show()

        model.train()
        model.fit(e, train_loader)
        
        model.eval()
        NMI, ARI, base_f1, base_auc, cac_f1, cac_auc = evaluate(model, test_loader)  # evaluation on the test_loader
        nmi_list.append(NMI)
        ari_list.append(ARI)
        
        print('Epoch: {:02d} | NMI: {:.3f} | ARI: {:.3f} | Base_F1: {:.3f} | Base_AUC: {:.3f} | CAC_F1: {:.3f} | CAC_F1: {:.3f}'.format(
            e+1, NMI, ARI, base_f1, base_auc, cac_f1, cac_auc))
        print("\n")

    return rec_loss_list, nmi_list, ari_list


def create_imbalanced_data_clusters(n_samples=10000, n_features=45, n_informative=35, n_classes=2,\
                            n_clusters = 2, frac=0.2, outer_class_sep=1.5, inner_class_sep=0.1, clus_per_class=2, seed=0):
    np.random.seed(seed)
    X = np.empty(shape=n_features)
    Y = np.empty(shape=1)
    offsets = np.random.normal(0, outer_class_sep, size=(n_clusters, n_features))
    for i in range(n_clusters):
        samples = int(np.random.normal(n_samples, n_samples/10))
        x, y = make_classification(n_samples=samples, n_features=n_features, n_informative=n_informative,\
                                    n_classes=n_classes, class_sep=inner_class_sep, n_clusters_per_class=clus_per_class)
                                    # n_repeated=0, n_redundant=0)
        x += offsets[i]
        y_0 = np.where(y == 0)[0]
        y_1 = np.where(y != 0)[0]
        y_1 = np.random.choice(y_1, int(np.random.normal(frac, frac/4)*len(y_1)))
        index = np.hstack([y_0,y_1])
        np.random.shuffle(index)
        x_new = x[index]
        y_new = y[index]

        X = np.vstack((X,x_new))
        Y = np.hstack((Y,y_new))

    X = X[1:,:]
    Y = Y[1:]
    return X, Y


def paper_synthetic(n_pts=1000, centers=4):
    X, y = make_blobs(n_pts, centers=centers)
    W = np.random.randn(10,2)
    U = np.random.randn(100,10)
    X1 = W.dot(X.T)
    X1 = X1*(X1>0)
    X2 = U.dot(X1)
    X2 = X2*(X2>0)
    return X2.T, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Clustering Network')

    # Dataset parameters
    parser.add_argument('--dir', default='../Dataset/sepsis/', 
                        help='dataset directory')
    parser.add_argument('--input-dim', type=int, default=89, 
                        help='input dimension')


    # Training parameters
    parser.add_argument('--lr', type=float, default=0.002, 
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--alpha', type=float, default=0.04, 
                        help='alpha (default: 4e-2)')
    parser.add_argument('--wd', type=float, default=5e-4, 
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=256, 
                        help='input batch size for training')
    parser.add_argument('--epoch', type=int, default=10, 
                        help='number of epochs to train')
    parser.add_argument('--pre-epoch', type=int, default=100, 
                        help='number of pre-train epochs')
    parser.add_argument('--pretrain', type=bool, default=True, 
                        help='whether use pre-training')
    

    # Model parameters
    parser.add_argument('--lamda', type=float, default=0.005,
                        help='coefficient of the reconstruction loss')
    parser.add_argument('--beta', type=float, default=1,
                        help='coefficient of the regularization term on ' \
                            'clustering')
    parser.add_argument('--hidden-dims', default=[500, 500, 2000],
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--latent-dim', type=int, default=45,
                        help='latent space dimension')
    parser.add_argument('--n-clusters', type=int, default=2,
                        help='number of clusters in the latent space')
    parser.add_argument('--clustering', type=str, default='cac', 
                        help='choose a clustering method (default: kmeans)' \
                       ' meanshift, tba')


    # Utility parameters
    parser.add_argument('--n-jobs', type=int, default=6,
                        help='number of jobs to run in parallel')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device for computation (default: cpu)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging the ' \
                            'training status')
    parser.add_argument('--test-run', action='store_true',
                        help='short test run on a few instances of the dataset')

    args = parser.parse_args()
    
    if args.dir != "../datasets/mnist":
        if args.dir == "../Dataset/sepsis/":
            X = pd.read_csv(args.dir + "X_new.csv")
            y = pd.read_csv(args.dir + "y_new.csv")

        elif args.dir == "synthetic":
            n_feat = 45
            X, y = create_imbalanced_data_clusters(n_samples=10000, n_clusters=2, n_features = n_feat, inner_class_sep=0.4, seed=0)
            X = pd.DataFrame(X)
            y = pd.DataFrame(y)
            args.input_dim = n_feat

        elif args.dir == "paper_synthetic":
            n_feat = 100
            X, y = paper_synthetic(10000)
            X = pd.DataFrame(X)
            y = pd.DataFrame(y)
            args.input_dim = n_feat

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        X_train_data_loader = list(zip(X_train.astype(np.float32), y_train.to_numpy().astype(np.float32).ravel()))
        X_test_data_loader  = list(zip(X_test.astype(np.float32), y_test.to_numpy().astype(np.float32).ravel()))

        train_loader = torch.utils.data.DataLoader(X_train_data_loader,
            batch_size=args.batch_size, shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(X_test_data_loader, 
            batch_size=args.batch_size, shuffle=False)

    elif args.dir == "../datasets/mnist":
        # Load data
        transformer = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,),
                                                               (0.3081,))])

        
        train_set = datasets.MNIST(args.dir, train=True, download=True, transform=transformer)
        test_set  = datasets.MNIST(args.dir, train=False, transform=transformer)
        train_limit = list(range(0, len(train_set))) if not args.test_run else list(range(0, 500))    
        test_limit  = list(range(0, len(test_set)))  if not args.test_run else list(range(0, 500))    

        
        train_loader = torch.utils.data.DataLoader(Subset(train_set, train_limit),
            batch_size=args.batch_size, shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(Subset(test_set, test_limit), 
            batch_size=args.batch_size, shuffle=False)

    # Main body
    model = DCN(args)    
    rec_loss_list, nmi_list, ari_list = solver(
        args, model, train_loader, test_loader)

    # X_train = X_train.to(self.device)
    # print(y_train[0])
    out = model.autoencoder(torch.FloatTensor(np.array(X_train)).to(args.device), latent=True)
    reducer = pacmap.PaCMAP()
    X2 = reducer.fit_transform(out.cpu().detach().numpy())
    X4 = reducer.fit_transform(X_train)
    c_train = [color[int(y_train.iloc[i])] for i in range(len(y_train))]

    figure = plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Normal vs CAC Embeddings')
    ax1.scatter(X4[:,0], X4[:,1], color=c_train)
    ax2.scatter(X2[:,0], X2[:,1], color=c_train)
    plt.savefig("normal_vs_cac.png", dpi=figure.dpi)
    # plt.show()

    # Testing
    out = model.autoencoder(torch.FloatTensor(np.array(X_test)).to(args.device), latent=True)
    reducer = pacmap.PaCMAP()
    # print(help(umap))
    X_t = reducer.fit_transform(out.cpu().detach().numpy())
    c_test = [color[int(y_test.iloc[i])] for i in range(len(y_test))]

    figure = plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('CAC Training vs Testing Embeddings')
    ax1.scatter(X2[:,0], X2[:,1], color=c_train)
    ax2.scatter(X_t[:,0], X_t[:,1], color=c_test)
    plt.savefig("train_vs_test.png", dpi=figure.dpi)
    # plt.show()
