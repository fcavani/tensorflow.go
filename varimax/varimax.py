#! /bin/env python3

import numpy as np
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
import h5py
import matplotlib
import matplotlib.pyplot as plt

def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old != 0 and d / d_old < 1 + tol: break
    return dot(Phi, R)

if __name__ == '__main__':
    with h5py.File("data_test.hdf5", "r") as f:
        # data = np.array(f['data_01'])
        pca = np.array(f['pca_test_01'])
        
    v = varimax(pca, gamma=1.0, q=30)
    _, ax = plt.subplots()
    ax.scatter(
        pca[:, 0],
        pca[:, 1],
        s=10,
        marker='o',
    )
    ax.scatter(
        v[:, 0],
        v[:, 1],
        s=10,
        marker='v',
    )
    plt.show()
