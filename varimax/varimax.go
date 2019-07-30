package varimax

import (
	"fmt"

	"github.com/fcavani/e"
	tfu "gitlab.com/fcavani/tensorflow.go/util"
	"gonum.org/v1/gonum/mat"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

type Varimax struct {
	Gamma float64
	Q     int
	Tol   float64
}

func NewVarimax() *Varimax {
	return &Varimax{
		Gamma: 1.0,
		Q:     20,
		Tol:   1e-6,
	}
}

func (v *Varimax) Fit(Phi *mat.Dense) (*mat.Dense, error) {
	root := op.NewScope()
	defer tfu.PanicRecover(root)

	p, k := Phi.Dims()
	phi := tfu.Dense2Matrix64(Phi)

	phit, err := tf.NewTensor(phi)
	if err != nil {
		return nil, e.New(err)
	}

	phphi := op.Placeholder(
		root.SubScope("phphi"),
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(int64(p), int64(k)),
		),
	)

	phLambda := op.Placeholder(
		root.SubScope("phLambda"),
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(int64(p), int64(k)),
		),
	)

	phu := op.Placeholder(
		root.SubScope("phu"),
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(int64(p), int64(k)),
		),
	)

	phvh := op.Placeholder(
		root.SubScope("phvh"),
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(int64(k), int64(k)),
		),
	)

	phR := op.Placeholder(
		root.SubScope("phR"),
		tf.Double,
		op.PlaceholderShape(
			tf.MakeShape(int64(k), int64(k)),
		),
	)

	// Make the graph
	R0 := R0Graph(root, int32(k))
	Lambda := LambdaGraph(root, phphi, phR)
	ops, opu, opvh := SVDGraph(root, phphi, phLambda, v.Gamma, float64(p))
	R := RGraph(root, phu, phvh)

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

	// Initial: R=eye(k)
	var results []*tf.Tensor
	results, err = sess.Run(map[tf.Output]*tf.Tensor{}, []tf.Output{R0}, nil)
	if err != nil {
		return nil, e.New(err)
	}
	r := results[0]

	var d, dold, tol float64
	var s, u, vh, lambda *tf.Tensor
	for i := 0; i < v.Q; i++ {
		dold = d

		//Lambda = dot(Phi, R)
		results, err = sess.Run(
			map[tf.Output]*tf.Tensor{phphi: phit, phR: r},
			[]tf.Output{Lambda},
			nil,
		)
		if err != nil {
			return nil, e.New(err)
		}
		lambda = results[0]

		//svd
		results, err = sess.Run(
			map[tf.Output]*tf.Tensor{phphi: phit, phLambda: lambda},
			[]tf.Output{ops, opu, opvh},
			nil,
		)
		if err != nil {
			return nil, e.New(err)
		}
		s = results[0]
		u = results[1]
		vh = results[2]

		//R = dot(u,vh)
		results, err = sess.Run(
			map[tf.Output]*tf.Tensor{phu: u, phvh: vh},
			[]tf.Output{R},
			nil,
		)
		if err != nil {
			return nil, e.New(err)
		}
		r = results[0]

		// d = sum(s)
		sm := s.Value().([]float64)
		d = 0
		for j := 0; j < len(sm); j++ {
			d += sm[j]
		}

		if i%1 == 0 {
			fmt.Printf("i: %v; tol: %.8f; d: %.12f\n", i, tol, d)
		}
		if dold != 0 && d/dold < 1.0+v.Tol {
			fmt.Printf("*i: %v; tol: %.8f; d: %.12f\n", i, tol, d)
			break
		}
	}

	return tfu.Matrix2Dense64(lambda.Value().([][]float64)), nil
}

// R0 make the graph for the initial R.
func R0Graph(root *op.Scope, k int32) tf.Output {
	//R = eye(k)
	// Inicialização de R
	r := op.Diag(
		root.SubScope("diag_R0"),
		op.Fill(
			root.SubScope("ones"),
			op.Const(
				root.SubScope("dims"),
				[1]int64{int64(k)},
			),
			op.Const(
				root.SubScope("def_value"),
				float64(1.0),
			),
		),
	)
	return r
}

//LambdaGraph make the graph for lambda.
func LambdaGraph(root *op.Scope, phphi, phR tf.Output) tf.Output {
	lambda := op.MatMul(
		root.SubScope("lambda"),
		phphi,
		phR,
	)
	return lambda
}

func SVDGraph(root *op.Scope, phphi, phLambda tf.Output, gamma, p float64) (tf.Output, tf.Output, tf.Output) {
	root = root.SubScope("svd")

	perm := op.Const(root.SubScope("perm"), [2]int64{1, 0})
	cov := op.MatMul(
		root.SubScope("cov"),
		op.Transpose(
			root.SubScope("transp_lambda"),
			phLambda,
			perm,
		),
		phLambda,
	)

	diag := op.Diag(
		root,
		op.DiagPart(
			root,
			cov,
		),
	)

	//u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
	ops, opu, opvh := op.Svd(
		root,
		op.MatMul(
			root.SubScope("main"),
			op.Transpose(
				root.SubScope("transp_phi"),
				phphi,
				perm,
			),
			op.Sub(
				root,
				op.Pow(
					root,
					phLambda,
					op.Const(
						root.SubScope("tres"),
						float64(3.0),
					),
				),
				op.Mul(
					root,
					op.MatMul(
						root.SubScope("lambda_diag"),
						phLambda,
						diag,
					),
					op.Div( // escalar
						root,
						op.Const(
							root.SubScope("gamma"),
							gamma,
						),
						op.Const(
							root.SubScope("p"),
							p,
						),
					),
				),
			),
		),
		op.SvdComputeUv(true),
		op.SvdFullMatrices(false),
	)
	return ops, opu, opvh
}

func RGraph(root *op.Scope, phu, phvh tf.Output) tf.Output {
	// R = dot(u,vh)
	// Update --> Update Lambda
	r := op.MatMul(
		root.SubScope("R"),
		phu,
		phvh,
	)
	return r
}
