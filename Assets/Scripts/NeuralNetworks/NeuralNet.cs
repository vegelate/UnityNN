﻿using System;

namespace NN
{
    public enum ActivationFunction { ReLU, Sigmoid }

    // 感知器
    [Serializable]
    public struct Perceptron
    {
        ActivationFunction activationFunction;
        public int LayerCount { get { return W.Length + 1; } }
        public Matrix[] W;  // 所有层的参数

        public Genoma GetGenoma { get { return new Genoma(W); } }
        
        // ctor
        // NeuronCount - 每一层维度的数组
        public Perceptron(Random r, int[] NeuronCount, ActivationFunction activationFunction)
        {
            this.activationFunction = activationFunction;

            W = new Matrix[NeuronCount.Length - 1];
            for (int i = 0; i < W.Length; i++)
            {
                int iNumRow = NeuronCount[i] + 1;
                int iNumCol = NeuronCount[i + 1];

                UnityEngine.Debug.Log("["+i+"] "+ iNumRow + " x " + iNumCol);

                W[i] = Matrix.Random(iNumRow, iNumCol, r) * 2 - 1;
            }
        }

        public Perceptron(Genoma genoma, ActivationFunction activationFunction)
        {
            this.activationFunction = activationFunction;
            W = genoma.W;
        }
        
        // 前向传播
        public Matrix ForwardPropagation(Matrix InputValue)
        {
            int m = InputValue.X;    // num of examples, rows
            var Z = new Matrix[LayerCount];
            var A = new Matrix[LayerCount];

            Z[0] = InputValue.AddColumn(Matrix.Ones(m, 1)); // add bias
            A[0] = Z[0];

            for (int i = 1; i < LayerCount; i++)
            {
                Z[i] = (A[i - 1] * W[i - 1]).AddColumn(Matrix.Ones(m, 1));
                A[i] = Activation(Z[i]);
            }
            var a = Z[Z.Length - 1];
            return a.Slice(0,1,a.X, a.Y);;
        }

        // 代价函数 y - 真实值，h - 预测值
        public static Matrix Cost(Matrix y, Matrix h)
        {
            return ((y - h).Pow(2.0) * 0.5).Sumatory(AxisZero.horizontal);
        }

        // 反向传播
        public void BackPropagation(Matrix m)
        {

        }

        Matrix Activation(Matrix m)
        {
            if (activationFunction == ActivationFunction.ReLU)
            {
                return Relu(m);
            }
            else if (activationFunction == ActivationFunction.Sigmoid)
            {
                return Sigmoid(m);
            }
            else
            {
                return null;
            }
        }

        Matrix Sigmoid(Matrix m)
        {
            double[,] output = m;
            Matrix.MatrixLoop((i, j) =>
            {
                output[i, j] = 1 / (1 + Math.Exp(-output[i, j]));

            }, m.X, m.Y);
            return output;
        }
        Matrix Relu(Matrix m)
        {
            double[,] output = m;
            Matrix.MatrixLoop((i, j) =>
            {
                output[i, j] = output[i, j] > 0 ? output[i, j] : 0;

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
                        m[i,j] += (r.NextDouble() * 2f - 1f) * maxPerturbation;
                    }
                }, gen.W[layer].X, gen.W[layer].Y);
                gen.W[layer] = m;
            }
            return gen;
        }
    }
}