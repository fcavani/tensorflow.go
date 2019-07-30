package util

import (
	"fmt"
	"io/ioutil"
	"os/exec"
	"runtime"
	"testing"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func Dense2Matrix32(x *mat.Dense) [][]float32 {
	var at float32
	row, cols := x.Caps()
	out := make([][]float32, row)
	for i := 0; i < row; i++ {
		outCol := make([]float32, cols)
		for j := 0; j < cols; j++ {
			at = float32(x.At(i, j))
			outCol[j] = at
		}
		out[i] = outCol
	}
	return out
}

func Dense2Matrix64(x *mat.Dense) [][]float64 {
	row, _ := x.Caps()
	out := make([][]float64, row)
	for i := 0; i < row; i++ {
		out[i] = x.RawRowView(i)
	}
	return out
}

func Matrix2Dense32(x [][]float32) *mat.Dense {
	rows := len(x)
	cols := len(x[0])
	out := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out.Set(i, j, float64(x[i][j]))
		}
	}
	return out
}

func Matrix2Dense64(x [][]float64) *mat.Dense {
	rows := len(x)
	cols := len(x[0])
	out := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		out.SetRow(i, x[i])
	}
	return out
}

func PlotPCAs32(res, proj [][]float32, t *testing.T) {
	m := len(res)
	if m != len(proj) {
		t.Fatal("projections sizes differ")
	}
	pts1 := make(plotter.XYs, 0, 10)
	for i := 0; i < m; i++ {
		var point struct{ X, Y float64 }
		point.X = float64(res[i][0])
		point.Y = float64(res[i][1])
		pts1 = append(pts1, point)
	}
	pts2 := make(plotter.XYs, 0, 10)
	for i := 0; i < m; i++ {
		var point struct{ X, Y float64 }
		point.X = float64(proj[i][0])
		point.Y = float64(proj[i][1])
		pts2 = append(pts2, point)
	}
	PlotPCAs(pts1, pts2, t)
}

func PlotPCAs64(res, proj [][]float64, t *testing.T) {
	m := len(res)
	if m != len(proj) {
		t.Fatal("projections sizes differ")
	}
	pts1 := make(plotter.XYs, 0, 10)
	for i := 0; i < m; i++ {
		var point struct{ X, Y float64 }
		point.X = res[i][0]
		point.Y = res[i][1]
		pts1 = append(pts1, point)
	}
	pts2 := make(plotter.XYs, 0, 10)
	for i := 0; i < m; i++ {
		var point struct{ X, Y float64 }
		point.X = proj[i][0]
		point.Y = proj[i][1]
		pts2 = append(pts2, point)
	}
	PlotPCAs(pts1, pts2, t)
}

func PlotPCAs(pts1, pts2 plotter.XYs, t *testing.T) {
	pl, err := plot.New()
	if err != nil {
		t.Fatal(err)
	}

	vs := []interface{}{}
	vs = append(vs, "tf")
	vs = append(vs, pts1)
	vs = append(vs, "pca")
	vs = append(vs, pts2)

	err = plotutil.AddScatters(pl, vs...)
	if err != nil {
		t.Fatal(err)
	}

	writeTo, err := pl.WriterTo(30*vg.Centimeter, 20*vg.Centimeter, "png")
	if err != nil {
		t.Fatal(err)
	}

	fd, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatal(err)
	}
	defer fd.Close()

	_, err = writeTo.WriteTo(fd)
	if err != nil {
		t.Fatal(err)
	}

	filename := fd.Name()
	fd.Close()

	cmd := exec.Command("open", filename)
	err = cmd.Run()
	if err != nil {
		t.Fatal(err)
	}
}

func Plot64(x []float64, t *testing.T) {
	pl, err := plot.New()
	if err != nil {
		t.Fatal(err)
	}

	pts := make(plotter.XYs, 0, 100)
	for i := 0; i < len(x); i++ {
		var point struct{ X, Y float64 }
		point.X = float64(i)
		point.Y = x[i]
		pts = append(pts, point)
	}

	vs := []interface{}{}
	vs = append(vs, "x")
	vs = append(vs, pts)

	err = plotutil.AddLines(pl, vs...)
	if err != nil {
		t.Fatal(err)
	}

	writeTo, err := pl.WriterTo(30*vg.Centimeter, 20*vg.Centimeter, "png")
	if err != nil {
		t.Fatal(err)
	}

	fd, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatal(err)
	}
	defer fd.Close()

	_, err = writeTo.WriteTo(fd)
	if err != nil {
		t.Fatal(err)
	}

	filename := fd.Name()
	fd.Close()

	cmd := exec.Command("open", filename)
	err = cmd.Run()
	if err != nil {
		t.Fatal(err)
	}
}

func RandomMatrix64(m, n int64) [][]float64 {
	root := op.NewScope()

	rsn := op.RandomStandardNormal(root,
		op.Const(root.SubScope("shape"), []int64{m, n}),
		tf.Double,
	)

	graph, err := root.Finalize()
	if err != nil {
		return nil
	}

	var sess *tf.Session
	sess, err = tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		return nil
	}
	defer sess.Close()

	var results []*tf.Tensor

	results, err = sess.Run(map[tf.Output]*tf.Tensor{}, []tf.Output{rsn}, nil)
	if err != nil {
		return nil
	}
	if len(results) != 1 {
		return nil
	}
	return results[0].Value().([][]float64)
}

func RandomMatrix32(m, n int64) [][]float32 {
	root := op.NewScope()

	rsn := op.RandomStandardNormal(root,
		op.Const(root.SubScope("shape"), []int64{m, n}),
		tf.Float,
	)

	graph, err := root.Finalize()
	if err != nil {
		return nil
	}

	var sess *tf.Session
	sess, err = tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		return nil
	}
	defer sess.Close()

	var results []*tf.Tensor

	results, err = sess.Run(map[tf.Output]*tf.Tensor{}, []tf.Output{rsn}, nil)
	if err != nil {
		return nil
	}
	if len(results) != 1 {
		return nil
	}
	return results[0].Value().([][]float32)
}

func MatrixEqual32(a, b [][]float32) bool {
	leni := len(a)
	if leni == 0 {
		return false
	}
	lenj := len(a[0])
	if lenj == 0 {
		return false
	}
	if leni != len(b) || lenj != len(b[0]) {
		return false
	}

	match := true
F:
	for i := 0; i < leni; i++ {
		for j := 0; j < lenj; j++ {
			if a[i][j] != b[i][j] {
				match = false
				break F
			}
		}
	}
	return match
}

func MatrixEqual64(a, b [][]float64) bool {
	leni := len(a)
	if leni == 0 {
		return false
	}
	lenj := len(a[0])
	if lenj == 0 {
		return false
	}
	if leni != len(b) || lenj != len(b[0]) {
		return false
	}

	match := true
F:
	for i := 0; i < leni; i++ {
		for j := 0; j < lenj; j++ {
			if a[i][j] != b[i][j] {
				match = false
				break F
			}
		}
	}
	return match
}

const recoverBufferStack = 4096

func PanicRecover(root *op.Scope) {
	if r := recover(); r != nil {
		fmt.Println(r)
		buf := make([]byte, recoverBufferStack)
		n := runtime.Stack(buf, false)
		buf = buf[:n]
		fmt.Printf("\n%v\n\n", string(buf))
		if err := root.Err(); err != nil {
			fmt.Printf("Scope Error:\n%v\n", err)
		}
	}
}
