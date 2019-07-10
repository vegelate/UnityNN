using System;

namespace NN
{
    public enum ActivationFunction { None, ReLU, Sigmoid, Linear, Tanh }

    // 感知器
    [Serializable]
    public struct NeuralNet
    {
        ActivationFunction[] activationFunctions;
        public int LayerCount { get { return W.Length; } }
        public Matrix[] W;  //  Weights
        public Matrix[] b;  //  Bias

        // ctor
        // NeuronCount - 每一层维度
        public NeuralNet(Random r, int[] NeuronCount, ActivationFunction[] activations)
        {
            this.activationFunctions = activations;

            W = new Matrix[NeuronCount.Length]; // W[i] 用于计算第 i 层的参数。 0 是输入层， W[0] 没有意义
            b = new Matrix[NeuronCount.Length];

            for (int i = 1; i < W.Length; i++)
            {
                int iNumRow = NeuronCount[i];
                int iNumCol = NeuronCount[i-1];

                if (activationFunctions[i] == ActivationFunction.ReLU)
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
                LinearActivationForward(A[i-1], W[i], b[i], this.activationFunctions[i], out Z[i], out A[i]);

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
        public void BackPropagation(Matrix y, Matrix h, ref Matrix[] Z, ref Matrix[] A, double learningRate, double lambda = 0.0)
        {
            double m = y.Y; // 训练数据条数
            double inv_m = 1.0 / m;

            Matrix[] dA = new Matrix[LayerCount];
            Matrix[] dW = new Matrix[LayerCount];
            Matrix[] db = new Matrix[LayerCount];

            dA[LayerCount - 1] = y - h;
            for (int iLayer = LayerCount-1; iLayer > 0; iLayer--)   // delta0 是输入层，不用计算
            {
                LinearActivationBackward(
                    dA[iLayer], Z[iLayer], W[iLayer], b[iLayer], A[iLayer - 1], this.activationFunctions[iLayer],
                    out dW[iLayer], out db[iLayer], out dA[iLayer-1]);
            }


            // 更新 W, b
            for (int i=1; i<W.Length; i++)
            {
                W[i] = W[i] + dW[i] * learningRate;
                b[i] = b[i] + db[i] * learningRate;

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
}