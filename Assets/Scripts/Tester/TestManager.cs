using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System;
using NN;

public class TestManager : MonoBehaviour
{
    delegate bool TestAction();

    struct stTester
    {
        public stTester(string _name, TestAction _action)
        {
            strName = _name;
            action = _action;
        }

        public string strName;
        public TestAction action;
    }

    List<stTester> m_listTester = new List<stTester>();

    // Start is called before the first frame update
    void Start()
    {
        _RegisterAllTester();

        if (UnityEditor.EditorApplication.isPlaying)
            StartCoroutine(_DoTest());
        else
            Debug.LogError("在运行状态下测试。");
    }

    // 执行测试
    IEnumerator _DoTest()
    {
        Debug.Log("<color=#00AA00ff>=================开始测试==================</color>");
        yield return null;

        int iPassCount = 0;
        for (int i = 0; i < m_listTester.Count; i++)
        {
            var tester = m_listTester[i];

            Debug.Log("<color=#00AA00ff>[" + i + "] " + tester.strName + " Testing... </color>");

            bool bRst = tester.action();

            if (bRst)
            {
                iPassCount++;
                Debug.Log("<color=#00AA00ff> [" + i + "] " + tester.strName + "---------- [Pass]</color>");
            }
            else
            {
                Debug.Log("<color=#CC0000ff> [" + i + "] " + tester.strName + "---------- [Fail]</color>");
            }
            yield return null;
        }

        Debug.Log("<color=white>测试完成，通过率 " + iPassCount + "/" + m_listTester.Count + "</color>");
    }

    // 
    void _RegisterAllTester()
    {
        m_listTester.Add(new stTester("Matrix", _TestMatrix));
        m_listTester.Add(new stTester("NN Foward", _TestNNFoward));
        m_listTester.Add(new stTester("Linear Backward", _TestLinearBackward));
        m_listTester.Add(new stTester("Linear Activation Backward", _TestLinearActivationBackward));
        m_listTester.Add(new stTester("NN Back", _TestNNBackPropagation));
    }

    bool _TestMatrix()
    {
        double[,] data1 = new double[5, 5];
        double[,] data2 = new double[5, 5];
        for (int x = 0; x < 5; x++)
        {
            for (int y = 0; y < 5; y++)
            {
                data1[x, y] = UnityEngine.Random.value;
                data2[x, y] = UnityEngine.Random.value;
            }
        }

        Matrix m1 = new Matrix(data1);
        Matrix m2 = new Matrix(data2);

        Matrix m3 = m1 + m2;

        Debug.Log(m1 + " + \n" + m2 + " = \n" + m3);


        return true;
    }

    bool _TestNNFoward()
    {

        Matrix A_prev = new Matrix(
            new double[3, 2]
            {
                { -0.41675785, - 0.05626683 },
                 { -2.1361961,   1.64027081 },
                 { -1.79343559, - 0.84174737 }
            });

        Matrix W = new Matrix(new double[1, 3] { { 0.50288142, -1.24528809, -1.05795222 } });
        Matrix b = new Matrix(new double[1, 1] { { -0.90900761 } });

        Matrix Z, A;
        NeuralNet.LinearActivationForward(A_prev, W, b, ActivationFunction.Sigmoid, out Z, out A);
        Debug.Log("Sigmoid A:" + A);

        NeuralNet.LinearActivationForward(A_prev, W, b, ActivationFunction.ReLU, out Z, out A);
        Debug.Log("Relu A:" + A);

        /*
        With sigmoid: A = [[0.96890023  0.11013289]]
        With ReLU: A = [[3.43896131  0.        ]]
        */
        return true;
    }

    // Linear backward
    bool _TestLinearBackward()
    {
        double[,] _A_prev = new double[3, 2]{
            {-0.52817175, -1.07296862},
            { 0.86540763, -2.3015387},
            {1.74481176 ,-0.7612069}};

        double[,] _W = new double[1, 3]
            {{0.3190391, -0.24937038, 1.46210794 }};

        double[,] _b = new double[1, 1] { { -2.06014071 } };

        double[,] _dZ = new double[1, 2] { { 1.62434536, -0.61175641 } };

        Matrix A_prev = new Matrix(_A_prev);
        Matrix W = new Matrix(_W);
        Matrix b = new Matrix(_b);
        Matrix dZ = new Matrix(_dZ);

        Matrix dW, dA_prev;
        Matrix db;

        NeuralNet.LinearBackward(dZ, W, b, A_prev, out dW, out db, out dA_prev);

        Debug.Log("dw:" + dW);
        Debug.Log("db:" + db);
        Debug.Log("dA_prev:" + dA_prev);


        /*
        dA_prev =  [[ 0.51822968 -0.19517421]
                     [-0.40506361  0.15255393]
                     [ 2.37496825 -0.89445391]]
        dW = [[-0.10076895  1.40685096  1.64992505]]
        db = [[0.50629448]]
        */
        return true;
    }


    bool _TestLinearActivationBackward()
    {
        Matrix dA = new Matrix(new double[1, 2]
        {
            { -0.41675785, -0.05626683 }
        });

        Matrix A_prev = new Matrix(new double[3, 2]
        {
            { -2.1361961, 1.64027081 },
            { -1.79343559, -0.84174737 },
            { 0.50288142, -1.24528809 }
         });

        Matrix W = new Matrix(new double[1, 3]
        {
            {-1.05795222, -0.90900761,  0.55145404}
        });

        Matrix b = new Matrix(new double[1,1] { { 2.29220801 } });

        Matrix Z = new Matrix(new double[1, 2]
        {
            {0.04153939, -1.11792545}
        });


        Matrix dW, db, dA_prev;
        
        NeuralNet.LinearActivationBackward(dA, Z, W, b, A_prev, ActivationFunction.Sigmoid, out dW, out db, out dA_prev);

        print("############# Sigmoid ############");
        print("dW:" + dW);
        print("db:" + db);
        print("dA_prev:" + dA_prev);
        
        print("############# Relu #############");
        NeuralNet.LinearActivationBackward(dA, Z, W, b, A_prev, ActivationFunction.ReLU, out dW, out db, out dA_prev);
        print("dW:" + dW);
        print("db:" + db);
        print("dA_prev:" + dA_prev);

        /*
        sigmoid:
        dA_prev = [[0.11017994  0.01105339]
                [0.09466817  0.00949723]
                [-0.05743092 - 0.00576154]]
        dW = [[0.10266786  0.09778551 - 0.01968084]]
        db = [[-0.05729622]]

        relu:
                dA_prev = [[0.44090989  0.        ]
                 [0.37883606  0.        ]
                 [-0.2298228   0.        ]]
        dW = [[0.44513824  0.37371418 - 0.10478989]]
        db = [[-0.20837892]]
*/
        return true;
    }

    // 反向传播
    bool _TestNNBackPropagation()
    {
        System.Random r = new System.Random(1);

        NeuralNet nn;
        Matrix x, y;    // training data

        // test relu
        if (false)
        {
            int m = 5;
            nn =
                new NeuralNet(r, new int[] { 1, 10, 1 }, ActivationFunction.ReLU);

            x = new Matrix(1, m);
            y = new Matrix(1, m);

            for (int i = 0; i < m; i++)
            {
                double x_i = (double)(i + 1);
                x.SetValue(0, i, x_i / 6.0);
                y.SetValue(0, i, x_i / 3.0);
            }
        }
        else
        // test sigmoid
        {
            int m = 2;
            nn =
                new NeuralNet(r, new int[] { 1, 3, 1 }, ActivationFunction.Sigmoid);

            x = new Matrix(1, m);
            y = new Matrix(1, m);

            x.SetValue(0, 0, -1.0);
            y.SetValue(0, 0, 0.0);

            x.SetValue(0, 1, 1.0);
            y.SetValue(0, 1, 1.0);
        }


        Debug.Log("x:" + x);
        Debug.Log("y:" + y);

        Matrix[] A;
        Matrix[] Z;
        Matrix h;
        for (int i = 0; i < 200; i++)
        {
            Debug.Log("==============" + i + "==============");
            h = nn.ForwardPropagation(x, out A, out Z);
            nn.BackPropagation(y, h, in Z, in A, 0.2);

            Debug.Log("h: " + h);

            var cost = NeuralNet.Cost(y, h);
            Debug.Log("cost:" + cost);
        }


        return true;
    }

    // Update is called once per frame
    void Update()
    {

    }
}
