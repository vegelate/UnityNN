using System;

namespace NN
{
    public enum ActivationFunction { ReLU, Sigmoid, Linear, Tanh }

    // 感知器
    [Serializable]
    public struct NeuralNet
    {
        ActivationFunction activationFunction;
        public int LayerCount { get { return W.Length; } }
        public Matrix[] W;  // 所有层的 Weights
        public Matrix[] b;  // 所有层的 bias
        public Genoma GetGenoma { get { return new Genoma(W); } }

        // ctor
        // NeuronCount - 每一层维度
        public NeuralNet(Random r, int[] NeuronCount, ActivationFunction activationFunction)
        {
            this.activationFunction = activationFunction;

            W = new Matrix[NeuronCount.Length]; // W[i] 用于计算第 i 层的参数。 0 是输入层， W[0] 没有意义
            b = new Matrix[NeuronCount.Length];

            for (int i = 1; i < W.Length; i++)
            {
                int iNumRow = NeuronCount[i];
                int iNumCol = NeuronCount[i-1];

                // UnityEngine.Debug.Log("[" + i + "] " + iNumRow + " x " + iNumCol);

                if (activationFunction == ActivationFunction.ReLU)
                {
                    W[i] = Matrix.Random(iNumRow, iNumCol, r);
                }
                else
                {
                    W[i] = Matrix.Random(iNumRow, iNumCol, r) * 2 - 1;
                }

                UnityEngine.Debug.Log("W[" + i + "]:" + W[i]);

                b[i] = new Matrix(iNumRow, 1);
            }


        }

        // 前向传播
        public Matrix ForwardPropagation(Matrix InputValue, out Matrix[] A, out Matrix[] Z)
        {
            int m = InputValue.Y;    // num of examples, rows
            Z = new Matrix[LayerCount];
            A = new Matrix[LayerCount];

            Z[0] = new Matrix(InputValue); 
            A[0] = new Matrix(InputValue);

            for (int i = 1; i < LayerCount; i++)
            {

                LinearActivationForward(A[i-1], W[i], b[i], this.activationFunction, out Z[i], out A[i]);

                UnityEngine.Debug.Log("A[" + (i-1) + "]:" + A[i-1]);
                UnityEngine.Debug.Log("W[" + (i) + "]:" + W[i]);
                UnityEngine.Debug.Log("b[" + (i) + "]:" + b[i]);
                UnityEngine.Debug.Log("Z[" + i + "]:" + Z[i]);
                UnityEngine.Debug.Log("A[" + i + "]:" + A[i]);
            }
            var a = A[A.Length - 1];
            return a;
        }

        // 代价函数 y - 真实值，h - 预测值
        public static Matrix Cost(Matrix y, Matrix h)
        {
            return ((y - h).Pow(2.0) * 0.5).Sumatory();
        }

        // 单次反向传播
        public void BackPropagation(Matrix y, Matrix h, in Matrix[] Z, in Matrix[] A, double learningRate, double lambda = 0.0)
        {
            double m = y.Y; // 训练数据条数
            double inv_m = 1.0 / m;

            Matrix[] dA = new Matrix[LayerCount];
            Matrix[] dW = new Matrix[LayerCount];
            Matrix[] db = new Matrix[LayerCount];

            //dA[LayerCount - 1] = y / h - (1.0 - y) / (1.0 - h);
            dA[LayerCount - 1] = y - h;
            for (int iLayer = LayerCount-1; iLayer > 0; iLayer--)   // delta0 是输入层，不用计算
            {
                LinearActivationBackward(
                    dA[iLayer], Z[iLayer], W[iLayer], b[iLayer], A[iLayer - 1], this.activationFunction,
                    out dW[iLayer], out db[iLayer], out dA[iLayer-1]);
            }

            for (int i=0; i<dA.Length; i++)
            {
                UnityEngine.Debug.Log("A[" + i + "] = " + A[i]);
                UnityEngine.Debug.Log("dA[" + i + "] = " + dA[i]);
            }

            // 更新 W, b
            for (int i=1; i<W.Length; i++)
            {
                W[i] = W[i] + dW[i] * learningRate;
                b[i] = b[i] + db[i] * learningRate;

                UnityEngine.Debug.Log("dW[" + i + "] = " + dW[i]);
                UnityEngine.Debug.Log("db[" + i + "] = " + db[i]);
            }

            { 
            /*
            double m = y.X; // 训练数据条数

            // 每层 a 的误差值
            var delta = new Matrix[LayerCount]; // dZ

            delta[LayerCount - 1] = y - h;   // 最后一层

            // 计算 delta
            for (int iLayer = W.Length - 1; iLayer > 0; iLayer--)   // delta0 是输入层，不用计算
            {

                Matrix d =  Matrix.ElementMult(
                         delta[iLayer + 1] * W[iLayer].T,
                         Matrix.ElementMult(A[iLayer], 1.0 - A[iLayer] ));//.RemoveColumn();

                delta[iLayer] = d.Slice(0, 1, d.X, d.Y);

                //UnityEngine.Debug.Log("delta[" + iLayer + "] " + delta[iLayer]);
            }

            for (int i = 1; i < delta.Length; i++)
            {
                UnityEngine.Debug.Log("delta[" + i + "]:" + delta[i]);
            }


            // 计算 Delta
            var Delta = new Matrix[W.Length];
            var grad = new Matrix[W.Length];    // 梯度

            for (int i = 0; i < W.Length; i++)
            {
                //Delta[i] = (delta[i + 1].T * A[i]).T;
                Delta[i] = A[i].T * delta[i+1];

                UnityEngine.Debug.Log("Delta[" + i + "]:" + Delta[i]);

                var reg = new Matrix(W[i]); // 正则化
                reg.SetRow(0, 0.0); // 第一列置零
                grad[i] = Delta[i] / m + reg * (lambda / m);

                UnityEngine.Debug.Log("grad " + i + ": " + grad[i]);
            }

            // 更新 W
            for (int i = 0; i < W.Length; i++)
            {
                UnityEngine.Debug.Log("before W " + i + ": " + W[i]);
                W[i] = W[i] + grad[i] * learningRate;
                UnityEngine.Debug.Log("after W " + i + ": " + W[i]);
            }

            */
            }
        }
        
        public static void LinearForward(Matrix A_prev, Matrix W, Matrix b, out Matrix Z)
        {
            Z = W * A_prev + b;
        }

        public static void LinearActivationForward(Matrix A_prev, Matrix W, Matrix b, ActivationFunction activationFunc, out Matrix Z, out Matrix A)
        {
            LinearForward(A_prev, W, b, out Z);
            A = Activation(Z, activationFunc);
        }

        public static void LinearActivationBackward(
            Matrix dA, Matrix Z, Matrix W, Matrix b, Matrix A_prev, ActivationFunction af, out Matrix dW, out Matrix db, out Matrix dA_prev)
        {
            Matrix dZ;
            if (af == ActivationFunction.Sigmoid)
                SigmoidBackward(dA, Z, out dZ);
            else if (af == ActivationFunction.ReLU)
                ReluBackward(dA, Z, out dZ);
            else
                dZ = Z;

            LinearBackward(dZ, W, b, A_prev, out dW, out db, out dA_prev);

        }

        // z = w*A_prev+b 的反向传播.
        // 已知dz, 求 dw, db, dA_prev 
        public static void LinearBackward(Matrix dZ, Matrix W, Matrix b, Matrix A_prev, out Matrix dW, out Matrix db, out Matrix dA_prev)
        {
            double m = A_prev.Y;
            double inv_m = 1.0 / m;

            dW = dZ * A_prev.T * inv_m;
            db = dZ.Sumatory(AxisZero.horizontal) * inv_m;

            UnityEngine.Debug.Log("W.T:" + W.T);
            UnityEngine.Debug.Log("dZ" + dZ);

            dA_prev = W.T * dZ;
        }

        public static void SigmoidBackward(Matrix dA, Matrix Z, out Matrix dZ)
        {
            Matrix s = Sigmoid(Z);
            dZ = Matrix.ElementMult(dA, Matrix.ElementMult(s, 1 - s));
        }

        public static void ReluBackward(Matrix dA, Matrix Z, out Matrix dZ)
        {
            var _dZ = new double[Z.X, Z.Y];
            double[,] _Z = Z;
            double[,] _dA = dA;
            Matrix.MatrixLoop((x, y) =>
            {
                _dZ[x, y] = _Z[x, y] < 0 ? 0 : _dA[x, y];
            }, Z.X, Z.Y);

            dZ = _dZ;
        }

        public static Matrix Activation(Matrix m, ActivationFunction activationFunction)
        {
            if (activationFunction == ActivationFunction.ReLU)
            {
                return Relu(m);
            }
            else if (activationFunction == ActivationFunction.Sigmoid)
            {
                return Sigmoid(m);
            }
            else if (activationFunction == ActivationFunction.Tanh)
            {
                return Tanh(m);
            }
            else if (activationFunction == ActivationFunction.Linear)
            {
                return m;
            }
            else
            {
                return null;
            }
        }

        static Matrix Sigmoid(Matrix m)
        {
            double[,] output = m;
            Matrix.MatrixLoop((i, j) =>
            {
                output[i, j] = 1 / (1 + Math.Exp(-output[i, j]));

            }, m.X, m.Y);
            return output;
        }
        static Matrix Relu(Matrix m)
        {
            double[,] output = m;
            Matrix.MatrixLoop((i, j) =>
            {
                output[i, j] = output[i, j] > 0 ? output[i, j] : 0;

            }, m.X, m.Y);
            return output;
        }
        static Matrix Tanh(Matrix m)
        {
            double[,] output = m;
            Matrix.MatrixLoop((i, j) =>
            {
                output[i, j] = Math.Tanh(output[i, j]);

            }, m.X, m.Y);
            return output;
        }
    }

    // 基因组
    [Serializable]
    public struct Genoma
    {
        public Matrix[] W;
        public Genoma(Matrix[] W)
        {
            this.W = W;
        }
        public static Genoma Cross(Random r, Genoma parent1, Genoma parent2)
        {
            Matrix[] SonW = new Matrix[parent1.W.Length];

            for (int layer = 0; layer < parent1.W.Length; layer++)
            {
                double[,] w = new double[parent1.W[layer].X, parent1.W[layer].Y];
                Matrix.MatrixLoop((i, j) =>
                {
                    if (r.NextDouble() > 0.5)
                    {
                        w[i, j] = parent1.W[layer].GetValue(i, j);
                    }
                    else
                    {
                        w[i, j] = parent2.W[layer].GetValue(i, j);
                    }
                }, parent1.W[layer].X, parent1.W[layer].Y);
                SonW[layer] = w;
            }

            return new Genoma(SonW);
        }
        public static Genoma Mutate(Random r, Genoma gen,
            float mutationRate, float maxPerturbation)
        {
            for (int layer = 0; layer < gen.W.Length; layer++)
            {
                double[,] m = gen.W[layer];
                Matrix.MatrixLoop((i, j) =>
                {
                    if (r.NextDouble() < mutationRate)
                    {
                        m[i, j] += (r.NextDouble() * 2f - 1f) * maxPerturbation;
                    }
                }, gen.W[layer].X, gen.W[layer].Y);
                gen.W[layer] = m;
            }
            return gen;
        }
    }
}