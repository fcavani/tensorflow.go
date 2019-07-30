# tensorflow.go
Tensorflow repo stores some machine learning algorithms developed in Go.

For now, there is only one algorithm working, the PCA, that is faster than the native
implementation for big matrices, but the flip algorithm doesn't work. The code is
pretty simple besides the Tensorflow part, with have some crazy matrix manipulations.
It's works more or less like the native go version
and the Scikit Learn implementations (python), the last that is the base reference for
this work.

I'm puting this on Github because I need some help to make it work like the python
version. My knowledge in TF is very new and TF is very complicated in Go at least.
Python version is much more easy to learn.

Fork it!

PS: I don't know varimax well, I think it's need more work.
