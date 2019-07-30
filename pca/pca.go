package math

import (
	"fmt"

	"github.com/fcavani/e"
	tfu "gitlab.com/fcavani/tensorflow.go/util"
	"gonum.org/v1/gonum/mat"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// PCA calc the PCA...
type PCA struct {
	components int
	Mean       float64
	S          []float64
	U          [][]float64
	V          [][]float64
}

// NewPCA creates a new PCA struct.
func NewPCA(components int) *PCA {
	p := &PCA{}
	p.components = components
	return p
}

// NumComponents return the number of resulting components.
func (p *PCA) NumComponents() (int, error) {
	if p.V == nil {
		return 0, e.New("need fit before transform")
	}
	if len(p.V) == 0 {
		return 0, e.New("v is empty")
	}
	if p.components > 0 {
		return p.components, nil
	}
	return len(p.V[1]), nil
}

func (p *PCA) Accumulated() (*mat.VecDense, error) {
	ratio, err := p.ExplainedVarianceRatio()
	if err != nil {
		return nil, e.Forward(err)
	}
	l := ratio.Len()
	accu := mat.NewVecDense(l, nil)
	accu.SetVec(0, ratio.AtVec(0))
	for i := 1; i < l; i++ {
		accu.SetVec(i, accu.AtVec(i-1)+ratio.AtVec(i))
	}
	return accu, nil
}

// ExplainedVarianceRatio is the percentage of variance explained
// by each of the selected components.
// If ``n_components`` is not set then all components are stored and the
// sum of explained variances is equal to 1.0.
func (p *PCA) ExplainedVarianceRatio() (*mat.VecDense, error) {
	if p.V == nil {
		return nil, e.New("need fit before transform")
	}

	numSamples := len(p.V)

	root := op.NewScope()
	defer tfu.PanicRecover(root)

	st, err := tf.NewTensor(p.S)
	if err != nil {
		return nil, e.New(err)
	}

	phs := op.Placeholder(
		root,
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(int64(len(p.S))),
		),
	)

	ev := ExplainedVarianceGraph(root, phs, numSamples)

	s := op.Sum(
		root,
		ev,
		op.Const(
			root.SubScope("axis0"),
			int64(0),
		),
	)

	ratio := op.Div(
		root.SubScope("ratio"),
		ev,
		s,
	)

	graph, err := root.Finalize()
	if err != nil {
		return nil, e.New(err)
	}

	var sess *tf.Session
	sess, err = tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		return nil, e.New(err)
	}
	defer sess.Close()

	var results []*tf.Tensor
	results, err = sess.Run(map[tf.Output]*tf.Tensor{phs: st}, []tf.Output{ratio}, nil)
	if err != nil {
		if er := root.Err(); er != nil {
			fmt.Println(err)
		}
		return nil, e.New(err)
	}

	if len(results) != 1 {
		return nil, e.New("run given wrong number of results")
	}

	num, err := p.NumComponents()
	if err != nil {
		return nil, e.Forward(err)
	}
	data := results[0].Value().([]float64)
	return mat.NewVecDense(num, data[:num]), nil
}

// ExplainedVariance is the amount of variance
// explained by each of the selected components.
func (p *PCA) ExplainedVariance() (*mat.VecDense, error) {
	//explained_variance_ = (S ** 2) / (n_samples - 1)
	if p.V == nil {
		return nil, e.New("need fit before transform")
	}

	numSamples := len(p.V)

	root := op.NewScope()
	defer tfu.PanicRecover(root)

	st, err := tf.NewTensor(p.S)
	if err != nil {
		return nil, e.New(err)
	}

	phs := op.Placeholder(
		root,
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(int64(len(p.S))),
		),
	)

	ev := ExplainedVarianceGraph(root, phs, numSamples)

	graph, err := root.Finalize()
	if err != nil {
		return nil, e.New(err)
	}

	var sess *tf.Session
	sess, err = tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		return nil, e.New(err)
	}
	defer sess.Close()

	var results []*tf.Tensor
	results, err = sess.Run(map[tf.Output]*tf.Tensor{phs: st}, []tf.Output{ev}, nil)
	if err != nil {
		if er := root.Err(); er != nil {
			fmt.Println(err)
		}
		return nil, e.New(err)
	}

	if len(results) != 1 {
		return nil, e.New("run given wrong number of results")
	}

	num, err := p.NumComponents()
	if err != nil {
		return nil, e.Forward(err)
	}
	data := results[0].Value().([]float64)
	return mat.NewVecDense(num, data[:num]), nil
}

func ExplainedVarianceGraph(root *op.Scope, phs tf.Output, numSamples int) tf.Output {
	root = root.SubScope("explained_variance")
	ev := op.Div(
		root.SubScope("ev"),
		op.Pow(
			root,
			phs,
			op.Const(
				root.SubScope("two"),
				float64(2.0),
			),
		),
		op.Const(
			root.SubScope("samples"),
			float64(numSamples-1),
		),
	)
	return ev
}

// Fit finds the svd and flip it.
func (p *PCA) Fit(dense *mat.Dense) (*PCA, error) {
	root := op.NewScope()
	defer tfu.PanicRecover(root)

	X := tfu.Dense2Matrix64(dense)

	xt, err := tf.NewTensor(X)
	if err != nil {
		return nil, e.New(err)
	}

	shape := xt.Shape()

	x := op.Placeholder(
		root,
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(shape[0], shape[1]),
		),
	)

	s, u, v, mean := SvdGraph(root, x)

	graph, err := root.Finalize()
	if err != nil {
		return nil, e.New(err)
	}

	var sess *tf.Session
	sess, err = tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		return nil, e.New(err)
	}
	defer sess.Close()

	var results []*tf.Tensor
	results, err = sess.Run(map[tf.Output]*tf.Tensor{x: xt}, []tf.Output{s, u, v, mean}, nil)
	if err != nil {
		if er := root.Err(); er != nil {
			fmt.Println(err)
		}
		return nil, e.New(err)
	}

	if len(results) != 4 {
		return nil, e.New("run given wrong number of results")
	}

	p.S = results[0].Value().([]float64)
	p.U = results[1].Value().([][]float64)
	p.V = results[2].Value().([][]float64)
	p.Mean = results[3].Value().(float64)

	return p, nil
}

// FitTransform approximate the pca like in fit and
// calculate the transformed matrix.
func (p *PCA) FitTransform(dense *mat.Dense) (*mat.Dense, error) {
	root := op.NewScope()
	defer tfu.PanicRecover(root)

	X := tfu.Dense2Matrix64(dense)

	xt, err := tf.NewTensor(X)
	if err != nil {
		return nil, e.New(err)
	}

	shape := xt.Shape()

	x := op.Placeholder(
		root,
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(shape[0], shape[1]),
		),
	)

	s, u, v, mean := SvdGraph(root, x)

	trans := TransGraph(root, x, mean, v, p.components)

	graph, err := root.Finalize()
	if err != nil {
		return nil, e.New(err)
	}

	var sess *tf.Session
	sess, err = tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		return nil, e.New(err)
	}
	defer sess.Close()

	var results []*tf.Tensor
	results, err = sess.Run(map[tf.Output]*tf.Tensor{x: xt}, []tf.Output{s, u, v, mean, trans}, nil)
	if err != nil {
		return nil, e.New(err)
	}

	if len(results) != 5 {
		return nil, e.New("run given wrong number of results")
	}

	p.S = results[0].Value().([]float64)
	p.U = results[1].Value().([][]float64)
	p.V = results[2].Value().([][]float64)
	p.Mean = results[3].Value().(float64)

	return tfu.Matrix2Dense64(results[4].Value().([][]float64)), nil
}

// Transform calculate the pca from the input matrix.
func (p *PCA) Transform(dense *mat.Dense) (*mat.Dense, error) {
	if p.V == nil {
		return nil, e.New("need fit before transform")
	}

	root := op.NewScope()
	defer tfu.PanicRecover(root)

	X := tfu.Dense2Matrix64(dense)

	xt, err := tf.NewTensor(X)
	if err != nil {
		return nil, e.New(err)
	}

	shape := xt.Shape()

	x := op.Placeholder(
		root.SubScope("phx"),
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(shape[0], shape[1]),
		),
	)

	vt, err := tf.NewTensor(p.V)
	if err != nil {
		return nil, e.New(err)
	}

	v := op.Placeholder(
		root.SubScope("phv"),
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(shape[1], shape[1]),
		),
	)

	meant, err := tf.NewTensor(p.Mean)
	if err != nil {
		return nil, e.New(err)
	}

	mean := op.Placeholder(
		root.SubScope("phm"),
		tf.Double,
		op.PlaceholderShape(tf.MakeShape(1, shape[1])),
	)

	trans := TransGraph(root, x, mean, v, p.components)

	graph, err := root.Finalize()
	if err != nil {
		return nil, e.New(err)
	}

	var sess *tf.Session
	sess, err = tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		return nil, e.New(err)
	}
	defer sess.Close()

	var results []*tf.Tensor
	results, err = sess.Run(map[tf.Output]*tf.Tensor{x: xt, v: vt, mean: meant}, []tf.Output{trans}, nil)
	if err != nil {
		return nil, e.New(err)
	}

	if len(results) != 1 {
		return nil, e.New("run given wrong number of results")
	}

	return tfu.Matrix2Dense64(results[0].Value().([][]float64)), nil
}

// SvdGraph implements the svd and mean to be used to calculate the pca.
func SvdGraph(root *op.Scope, x tf.Output) (tf.Output, tf.Output, tf.Output, tf.Output) {
	root = root.SubScope("svd")
	mean := op.Mean(root.SubScope("mean"),
		x,
		op.Const(root.SubScope("axes"), [2]int64{0, 1}),
	)

	xm := op.Sub(root,
		x,
		mean,
	)
	// s is a tensor of singular values for each matrix.
	// u is the tensor containing of left singular vectors for each matrix
	// v is the tensor containing of right singular vectors for each matrix.
	s, u, v := op.Svd(
		root,
		xm,
		op.SvdComputeUv(true),
		op.SvdFullMatrices(false),
	)

	sign := SvdSignU(root, u)
	u = SvdFlipU(root, sign, u)
	v = SvdFlipV(root, sign, v)

	return s, u, v, mean
}

// TransGraph is the sub graph to approximate the pca.
func TransGraph(root *op.Scope, x, mean, v tf.Output, comps int) tf.Output {
	root = root.SubScope("svd_trans")
	xm := op.Sub(root,
		x,
		mean,
	)
	if comps > 0 {
		shape, _ := v.Shape().ToSlice()
		v = op.Slice(
			root.SubScope("slice_v"),
			v,
			op.Const(root.SubScope("begin"), [2]int64{0, 0}),
			op.Const(root.SubScope("begin"), [2]int64{shape[0], int64(comps)}),
		)
	}
	pca := op.BatchMatMul(root.SubScope("multi_x_eig"),
		xm,
		v,
	)
	return pca
}

// SvdSignU calculate the matrix with the signals from u.
func SvdSignU(scope *op.Scope, u tf.Output) tf.Output {
	// columns of u, rows of v
	scope = scope.SubScope("svd_sign_u")

	abs := op.Abs(
		scope,
		u,
	)

	argmax := op.ArgMax(
		scope,
		abs,
		op.Const(scope.SubScope("dim"), int64(0)),
	)

	shape, _ := abs.Shape().ToSlice()

	idxHot := op.OneHot(
		scope,
		argmax,
		op.Const(scope.SubScope("shape_one_hot"),
			int32(shape[0]),
		),
		op.Const(scope.SubScope("on_one_hot"),
			1.0,
		),
		op.Const(scope.SubScope("off_one_hot"),
			0.0,
		),
	)

	perm := op.Const(
		scope.SubScope("permT"),
		[2]int64{1, 0},
	)

	idxHot = op.Transpose(
		scope.SubScope("idxHotT"),
		idxHot,
		perm,
	)

	sign := op.Sign(
		scope,
		u,
	)

	sign = op.Mul(
		scope.SubScope("sign_mul"),
		sign,   // negatives
		idxHot, //max
	)

	zeros := op.Equal(
		scope,
		sign,
		op.Const(
			scope.SubScope("zero"),
			float64(0.0),
		),
	)

	sign = op.Add(
		scope.SubScope("sign_mul2"),
		sign,
		op.Cast(
			scope,
			zeros,
			tf.Double,
		),
	)

	signProdMat := op.Prod(
		scope,
		sign,
		op.Const(
			scope.SubScope("axis_sum"),
			int64(0),
		),
	)

	signProdMat = op.Reshape(
		scope,
		signProdMat,
		op.Const(
			scope.SubScope("x"),
			[2]int64{shape[1], 1},
		),
	)

	return signProdMat
}

// SvdFlipU flips the u matrix from svd.
func SvdFlipU(scope *op.Scope, signProdMat, u tf.Output) tf.Output {
	perm := op.Const(
		scope.SubScope("permT"),
		[2]int64{1, 0},
	)
	signProdMatT := op.Transpose(
		scope.SubScope("signSumMatT"),
		signProdMat,
		perm,
	)

	u = op.Mul(
		scope.SubScope("mul_u"),
		u,
		signProdMatT,
	)

	return u
}

// SvdFlipV flips the v matrix from svd.
func SvdFlipV(scope *op.Scope, signProdMat, v tf.Output) tf.Output {
	v = op.Mul(
		scope.SubScope("mul_u"),
		v,           // n x n
		signProdMat, //m x 1
	)
	return v // n x n
}
