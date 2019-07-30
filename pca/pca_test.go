package math

import (
	"testing"

	"github.com/sjwhitworth/golearn/pca"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	tfu "gitlab.com/fcavani/tensorflow.go/util"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/hdf5"
)

func TestSVDFlipU(t *testing.T) {
	const m = 3
	const n = 4

	u := [][]float64{
		{0.0, 0.0, 0.0, 5.0},
		{1.0, 0.0, 43.0, 3.0},
		{-24.0, 0.0, 5.0, -6.0},
	}

	wanted := [][]float64{
		{0.0, 0.0, 0.0, -5.0},
		{-1.0, 0.0, 43.0, -3.0},
		{24.0, 0.0, 5.0, 6.0},
	}

	root := op.NewScope()
	defer tfu.PanicRecover(root)

	ut, err := tf.NewTensor(u)
	if err != nil {
		t.Fatal(err)
	}

	phu := op.Placeholder(
		root.SubScope("phu"),
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(m, n),
		),
	)

	sign := SvdSignU(root, phu)
	newU := SvdFlipU(root, sign, phu)

	graph, err := root.Finalize()
	if err != nil {
		t.Fatal(err)
	}

	var sess *tf.Session
	sess, err = tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	var results []*tf.Tensor
	results, err = sess.Run(map[tf.Output]*tf.Tensor{phu: ut}, []tf.Output{newU}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) != 1 {
		t.Fatal("run given wrong number of results")
	}

	got := results[0].Value().([][]float64)

	t.Log(results[0].Value().([][]float64))

	if !tfu.MatrixEqual64(wanted, got) {
		t.Log("invalid u")
	}
}

func TestSVDFlipV(t *testing.T) {
	const m = 3
	const n = 4

	u := [][]float64{
		{0.0, 0.0, 0.0, 5.0},
		{1.0, 0.0, 43.0, 3.0},
		{-24.0, 0.0, 5.0, -6.0},
	}

	v := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}

	wanted := [][]float64{
		{-1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, -1},
	}

	root := op.NewScope()
	defer tfu.PanicRecover(root)

	ut, err := tf.NewTensor(u)
	if err != nil {
		t.Fatal(err)
	}

	phu := op.Placeholder(
		root.SubScope("phu"),
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(m, n),
		),
	)

	vt, err := tf.NewTensor(v)
	if err != nil {
		t.Fatal(err)
	}

	phv := op.Placeholder(
		root.SubScope("phu"),
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(n, n),
		),
	)

	sign := SvdSignU(root, phu)
	newV := SvdFlipV(root, sign, phv)

	graph, err := root.Finalize()
	if err != nil {
		t.Fatal(err)
	}

	var sess *tf.Session
	sess, err = tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	var results []*tf.Tensor
	results, err = sess.Run(map[tf.Output]*tf.Tensor{phu: ut, phv: vt}, []tf.Output{newV}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) != 1 {
		t.Fatal("run given wrong number of results")
	}

	got := results[0].Value().([][]float64)

	t.Log(results[0].Value().([][]float64))

	if !tfu.MatrixEqual64(wanted, got) {
		t.Log("invalid v")
	}
}

func TestPCAFitTransform(t *testing.T) {
	f, err := hdf5.OpenFile("data_test.hdf5", hdf5.F_ACC_RDONLY)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	data01, err := f.OpenDataset("data_01")
	if err != nil {
		t.Fatal(err)
	}
	defer data01.Close()
	pca01, err := f.OpenDataset("pca_test_01")
	if err != nil {
		t.Fatal(err)
	}
	defer pca01.Close()

	X1 := make([]float64, 21)
	Xt := make([]float64, 21)

	err = data01.Read(&X1)
	if err != nil {
		t.Fatal(err)
	}
	err = pca01.Read(&Xt)
	if err != nil {
		t.Fatal(err)
	}

	denseX1 := mat.NewDense(7, 3, X1)
	denseXt := mat.NewDense(7, 3, Xt)

	p := NewPCA(0)
	pa, err := p.FitTransform(denseX1)
	if err != nil {
		t.Fatal(err)
	}
	proja := tfu.Dense2Matrix64(pa)

	paflat := mat.NewDense(21, 1, pa.RawMatrix().Data)
	Xtflat := mat.NewDense(21, 1, denseXt.RawMatrix().Data)

	paflat.Sub(paflat, Xtflat)
	paflat.MulElem(paflat, paflat)
	mse := mat.Sum(paflat) / float64(21.0)

	t.Logf("Error (mse): %.4f\n", mse)

	if !tfu.MatrixEqual64(proja, tfu.Dense2Matrix64(denseXt)) {
		t.Log("tf projection didn't mach with python projection")
	}

	// tfu.PlotPCAs64(proja, tfu.Dense2Matrix64(denseXt), t)
}

func TestPCAFitAndTransform(t *testing.T) {
	f, err := hdf5.OpenFile("data_test.hdf5", hdf5.F_ACC_RDONLY)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	data01, err := f.OpenDataset("data_01")
	if err != nil {
		t.Fatal(err)
	}
	defer data01.Close()
	pca01, err := f.OpenDataset("pca_test_01")
	if err != nil {
		t.Fatal(err)
	}
	defer pca01.Close()

	X1 := make([]float64, 21)
	Xt := make([]float64, 21)

	err = data01.Read(&X1)
	if err != nil {
		t.Fatal(err)
	}
	err = pca01.Read(&Xt)
	if err != nil {
		t.Fatal(err)
	}

	denseX1 := mat.NewDense(7, 3, X1)
	denseXt := mat.NewDense(7, 3, Xt)

	p := NewPCA(0)
	_, err = p.Fit(denseX1)
	if err != nil {
		t.Fatal(err)
	}
	pa, err := p.Transform(denseX1)
	if err != nil {
		t.Fatal(err)
	}
	// pa, err := p.FitTransform(X1)
	// if err != nil {
	// 	t.Fatal(err)
	// }
	proja := tfu.Dense2Matrix64(pa)

	paflat := mat.NewDense(21, 1, pa.RawMatrix().Data)
	Xtflat := mat.NewDense(21, 1, denseXt.RawMatrix().Data)

	paflat.Sub(paflat, Xtflat)
	paflat.MulElem(paflat, paflat)
	mse := mat.Sum(paflat) / float64(21.0)

	t.Logf("Error (mse): %.4f\n", mse)

	if !tfu.MatrixEqual64(proja, tfu.Dense2Matrix64(denseXt)) {
		t.Log("tf projection didn't mach with python projection")
	}

	// tfu.PlotPCAs64(proja, tfu.Dense2Matrix64(denseXt), t)
}

func TestPCAFitTransformRepetition(t *testing.T) {
	const numTests = 10

	X1 := mat.NewDense(7, 3, []float64{6, 5, 4, 3, 8, 2, 9, 5, 1, 10, 2, 3, 8, 7, 5, 14, 2, 3, 6, 3, 2})

	results := make([][][]float64, numTests)

	for i := 0; i < numTests; i++ {
		p := &PCA{}
		_, err := p.Fit(X1)
		if err != nil {
			t.Fatal(err)
		}
		pa, err := p.Transform(X1)
		if err != nil {
			t.Fatal(err)
		}
		results[i] = tfu.Dense2Matrix64(pa)
	}

	for i := 1; i < numTests; i++ {
		if !tfu.MatrixEqual64(results[0], results[i]) {
			t.Log("tf projection didn't mach with go projection:", i)
		}
	}
}

func TestPCAFitAndTransformComponents(t *testing.T) {
	const m = 30
	const n = 10

	dense := tfu.Matrix2Dense64(tfu.RandomMatrix64(m, n))

	p := NewPCA(5)
	_, err := p.Fit(dense)
	if err != nil {
		t.Fatal(err)
	}
	pa, err := p.Transform(dense)
	if err != nil {
		t.Fatal(err)
	}

	x, y := pa.Caps()
	if x != m {
		t.Fatal("invalid size, m:", x)
	}
	if y != 5 {
		t.Fatal("invalid size, n:", y)
	}

}

func TestNumComponents(t *testing.T) {
	const m = 30
	const n = 10

	dense := tfu.Matrix2Dense64(tfu.RandomMatrix64(m, n))

	p := NewPCA(0)
	_, err := p.Fit(dense)
	if err != nil {
		t.Fatal(err)
	}

	num, err := p.NumComponents()
	if err != nil {
		t.Fatal(err)
	}
	if num != 10 {
		t.Fatal("wrong number of components")
	}

	p = NewPCA(5)
	_, err = p.Fit(dense)
	if err != nil {
		t.Fatal(err)
	}

	num, err = p.NumComponents()
	if err != nil {
		t.Fatal(err)
	}
	if num != 5 {
		t.Fatal("wrong number of components")
	}
}

func TestExplainedVarianceRatio(t *testing.T) {
	f, err := hdf5.OpenFile("data_test.hdf5", hdf5.F_ACC_RDONLY)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	data01, err := f.OpenDataset("data_01")
	if err != nil {
		t.Fatal(err)
	}
	defer data01.Close()

	X1 := make([]float64, 21)

	err = data01.Read(&X1)
	if err != nil {
		t.Fatal(err)
	}

	denseX1 := mat.NewDense(7, 3, X1)

	p := NewPCA(0)
	_, err = p.FitTransform(denseX1)
	if err != nil {
		t.Fatal(err)
	}

	evr, err := p.ExplainedVarianceRatio()
	if err != nil {
		t.Fatal(err)
	}

	s := mat.Sum(evr)
	if s < 0.999 {
		t.Fatal("sum failed", s)
	}
}

func TestExplainedVarianceAccumulated(t *testing.T) {
	const m = 1000
	const n = 10

	dense := tfu.Matrix2Dense64(tfu.RandomMatrix64(m, n))

	p := NewPCA(0)
	_, err := p.FitTransform(dense)
	if err != nil {
		t.Fatal(err)
	}

	accu, err := p.Accumulated()
	if err != nil {
		t.Fatal(err)
	}

	l := accu.Len()
	out := make([]float64, l)
	for i := 0; i < l; i++ {
		out[i] = accu.AtVec(i)
	}

	tfu.Plot64(out, t)
}

const sizeMatrixM = 500
const sizeMatrixN = 500

func BenchmarkPCAgolang64(b *testing.B) {
	data := tfu.Matrix2Dense64(tfu.RandomMatrix64(sizeMatrixM, sizeMatrixN))
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		p := pca.NewPCA(0)
		p.FitTransform(data)
	}
}

func BenchmarkPCAtensorflow64(b *testing.B) {
	data := tfu.Matrix2Dense64(tfu.RandomMatrix64(sizeMatrixM, sizeMatrixN))
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		p := NewPCA(0)
		p.FitTransform(data)
	}
}
