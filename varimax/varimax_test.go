package varimax

import (
	"testing"

	tfu "gitlab.com/fcavani/tensorflow.go/util"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/hdf5"
)

func TestVarimax(t *testing.T) {
	f, err := hdf5.OpenFile("data_test.hdf5", hdf5.F_ACC_RDONLY)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	pca01, err := f.OpenDataset("pca_test_01")
	if err != nil {
		t.Fatal(err)
	}
	defer pca01.Close()

	X1 := make([]float64, 21)

	err = pca01.Read(&X1)
	if err != nil {
		t.Fatal(err)
	}

	denseX1 := mat.NewDense(7, 3, X1)

	vm := NewVarimax()
	vm.Gamma = 1.0
	vm.Q = 30
	v, err := vm.Fit(denseX1)
	if err != nil {
		t.Fatal(err)
	}
	// t.Log(v)

	vdense := tfu.Dense2Matrix64(v)
	X1m := tfu.Dense2Matrix64(denseX1)

	tfu.PlotPCAs64(X1m, vdense, t)
}
