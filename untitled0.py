import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import time
import scipy.linalg
import scipy.sparse.linalg
from sklearn.utils.extmath import _randomized_eigsh
from kernels import kernel_factory
from scipy.optimize import minimize
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
TensorType = torch.DoubleTensor
torch.set_default_tensor_type(TensorType)
import torch
import time
import scipy.linalg
from dc import KPCA_DC, RobustKPCA_DC, SparseKPCA_DC
from kernels import kernel_factory

def center_train_gram_matrix(Kxx):
    '''
    Input:
        Kxx: numpy array of shape (n, n)
    Output:
        Kxx_c: numpy array of shape (n, n)
    '''
    n = Kxx.shape[0]
    one_n_n = torch.ones((n, n)) / n
    one_matrix = torch.ones((n,n)) / n
    Kxx_c = Kxx - Kxx @ one_matrix - one_matrix @ Kxx + one_matrix @ Kxx @ one_matrix
    return Kxx_c

random_state = 17
n_samples = 4000
X, y = make_moons(n_samples=n_samples, noise=0.01, random_state=random_state)
#X, y = make_circles(n_samples=n_samples, noise=0.01, factor=0.3, random_state=random_state)
#X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=random_state)

fig, ax = plt.subplots()
_ = ax.scatter(X[:,0], X[:,1], c=y)
ax.set_aspect('equal')

kernel = kernel_factory("rbf", {"sigma2": 1.0})
X = torch.from_numpy(X).double()
#X = X - X.mean(axis=0)
G = kernel(X.t())
G = center_train_gram_matrix(G)
# %%
s = 2
model = KPCA_DC(s)
torch.manual_seed(10)
model.fit2(G)
# %%
H = model.H
# %%
print("Convergence reached:", model.exit_code == 0)
model2 = KPCA_DC(s)
torch.manual_seed(10)
model2.fit(G)