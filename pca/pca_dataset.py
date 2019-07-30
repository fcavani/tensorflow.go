#! /bin/env python3

import numpy as np
import h5py
from sklearn.decomposition import PCA

x1 = np.array([6, 5, 4, 3, 8, 2, 9, 5, 1, 10, 2, 3, 8, 7, 5, 14, 2, 3, 6, 3, 2], dtype=np.float)
x1 = x1.reshape((7,3))

pca1 = PCA(n_components=3)
t1 = pca1.fit_transform(x1)

with h5py.File("data_test.hdf5", "w") as f:
    f.create_dataset("data_01", data=x1.astype('float64').data)
    f.create_dataset("pca_test_01", data=t1.astype('float64').data)
    # dset = f.create_dataset("pca_test_01", data=t1)
    # dset.flush()
