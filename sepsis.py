import torch
import argparse
import numpy as np
import umap
from DCN import DCN
from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import Subset
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']

def evaluate(model, test_loader):
    y_test = []
    y_pred = []
    for data, target in test_loader:
        batch_size = data.size()[0]
        data = data.view(batch_size, -1).to(model.device)
        latent_X = model.autoencoder(data, latent=True)
        latent_X = latent_X.detach().cpu().numpy()

        y_test.append(target.view(-1, 1).numpy())
        y_pred.append(model.clustering.update_assign(latent_X).reshape(-1, 1))
    
    y_test = np.vstack(y_test).reshape(-1)
    y_pred = np.vstack(y_pred).reshape(-1)
    return (normalized_mutual_info_score(y_test, y_pred),
            adjusted_rand_score(y_test, y_pred))


def solver(args, model, train_loader, test_loader):
    rec_loss_list = model.pretrain(train_loader, epoch=args.pre_epoch)
    nmi_list = []
    ari_list = []

    for e in range(args.epoch):
        model.train()
        model.fit(e, train_loader)
        
        model.eval()
        NMI, ARI = evaluate(model, test_loader)  # evaluation on the test_loader
        nmi_list.append(NMI)
        ari_list.append(ARI)
        
        print('Epoch: {:02d} | NMI: {:.3f} | ARI: {:.3f}'.format(
            e+1, NMI, ARI))

    return rec_loss_list, nmi_list, ari_list


def create_imbalanced_data_clusters(n_samples=1000, n_features=8, n_informative=5, n_classes=2,\
                            n_clusters = 3, frac=0.2, outer_class_sep=1, inner_class_sep=0.5, clus_per_class=2, seed=0):
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
    parser.add_argument('--pre-epoch', type=int, default=5, 
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
    parser.add_argument('--latent-dim', type=int, default=10,
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
    parser.add_argument('--log-interval', type=int, default=100,
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
            X, y = create_imbalanced_data_clusters(seed=0)
            X = pd.DataFrame(X)
            y = pd.DataFrame(y)
            args.input_dim = 8

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
    out = model.autoencoder(torch.FloatTensor(np.array(X_train)), latent=True)
    reducer = umap.UMAP()
    # print(help(umap))
    X2 = reducer.fit_transform(out.detach().numpy())
    c = [color[int(y_train.iloc[i])] for i in range(len(y_train))]
    plt.scatter(X2[:,0], X2[:,1], color=c); plt.show()

    X4 = reducer.fit_transform(X_train)
    plt.scatter(X4[:,0], X4[:,1], color=c); plt.show()