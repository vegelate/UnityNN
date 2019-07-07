﻿using System;

namespace NN
{
    public enum ActivationFunction { ReLU, Sigmoid, Linear, Tanh }

    // 感知器
    [Serializable]
    public struct NeuralNet
    {
        ActivationFunction activationFunction;
        public int LayerCount { get { return W.Length + 1; } }
        public Matrix[] W;  // 所有层的 Weights
        public double[] b;  // 所有层的 bias
        public Genoma GetGenoma { get { return new Genoma(W); } }

        // ctor
        // NeuronCount - 每一层维度
        public NeuralNet(Random r, int[] NeuronCount, ActivationFunction activationFunction)
        {
            this.activationFunction = activationFunction;

            W = new Matrix[NeuronCount.Length - 1];

            for (int i = 0; i < W.Length; i++)
            {
                int iNumRow = NeuronCount[i];
                int iNumCol = NeuronCount[i + 1];

                // UnityEngine.Debug.Log("[" + i + "] " + iNumRow + " x " + iNumCol);

                W[i] = Matrix.Random(iNumRow, iNumCol, r) * 2 - 1;
                UnityEngine.Debug.Log("W[" + i + "]:" + W[i]);
            }

            b = new double[NeuronCount.Length - 1];
        }

        // 前向传播
        public Matrix ForwardPropagation(Matrix InputValue, out Matrix[] A)
        {
            int m = InputValue.X;    // num of examples, rows
            var Z = new Matrix[LayerCount];
            A = new Matrix[LayerCount];

            Z[0] = InputValue; // add bias
            A[0] = Z[0];

            for (int i = 1; i < LayerCount; i++)
            {
                Z[i] = (A[i - 1] * W[i - 1]) + b[i - 1]; // z = w*A + b
                A[i] = Activation(Z[i]);


                UnityEngine.Debug.Log("A[" + (i - 1) + "]:" + A[i - 1]);
                UnityEngine.Debug.Log("W[" + (i - 1) + "]:" + W[i - 1]);
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
        public void BackPropagation(Matrix y, Matrix h, in Matrix[] A, double learningRate, double lambda = 0.0)
        {
            double m = y.X; // 训练数据条数
            double inv_m = 1.0 / m;

            Matrix[] dA = new Matrix[LayerCount];
            Matrix[] dW = new Matrix[LayerCount];
            double[] db = new double[LayerCount];

            dA[LayerCount - 1] = y - h;
            for (int iLayer = W.Length - 1; iLayer > 0; iLayer--)   // delta0 是输入层，不用计算
            {
                LinearActivationBackward(
                    dA[iLayer+1], A[iLayer], W[iLayer], b[iLayer], A[iLayer - 1], 
                    out dW[iLayer], out db[iLayer], out dA[iLayer-1]);
              
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
        
        public static void LinearForward(Matrix A_prev, Matrix W, double b, out Matrix Z)
        {
            Z = W * A_prev + b;
        }

        public static void LinearActivationForward(Matrix A_prev, Matrix W, double b, ActivationFunction activationFunc, out Matrix Z, out Matrix A)
        {
            LinearForward(A_prev, W, b, out Z);
            A = Activation(Z, activationFunc);
        }

        void LinearActivationBackward(Matrix dA, Matrix A, Matrix W, double b, Matrix A_prev, out Matrix dW, out double db, out Matrix dA_prev)
        {
            Matrix dZ;
            SigmoidBackward(dA, A, out dZ);
            LinearBackward(dZ, W, b, A_prev, out dW, out db, out dA_prev);

        }

        // z = w*A_prev+b 的反向传播.
        // 已知dz, 求 dw, db, dA_prev 
        public static void LinearBackward(Matrix dZ, Matrix W, double b, Matrix A_prev, out Matrix dW, out double db, out Matrix dA_prev)
        {
            double m = A_prev.Y;
            double inv_m = 1.0 / m;

            dW = dZ * A_prev.T * inv_m;
            db = dZ.Sumatory().GetValue(0, 0) * inv_m;
            dA_prev = W.T * dZ;
        }

        void SigmoidBackward(Matrix dA, Matrix A, out Matrix dZ)
        {
            dZ = Matrix.ElementMult(dA, Matrix.ElementMult(A, 1 - A));
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