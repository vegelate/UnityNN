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
        Debug.Log("=================开始测试==================");
        yield return null;

        int iPassCount = 0;
        for (int i = 0; i < m_listTester.Count; i++)
        {
            var tester = m_listTester[i];

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
        //m_listTester.Add(new stTester("NN Back", _TestNNBackPropagation));
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
        System.Random r = new System.Random(1);

        NeuralNet p =
            new NeuralNet(r, new int[] { 2, 4, 8 }, ActivationFunction.ReLU);

        Matrix input = Matrix.Random(1, 2, r);

        Matrix[] A;
        Matrix result = p.ForwardPropagation(input, out A);

        Debug.Log("Result:" + result);

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

        double b = -2.06014071;

        double[,] _dZ = new double[1, 2] { { 1.62434536, -0.61175641 } };

        Matrix A_prev = new Matrix(_A_prev);
        Matrix W = new Matrix(_W);
        Matrix dZ = new Matrix(_dZ);

        Matrix dW, dA_prev;
        double db;

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

    bool _TestActivationBackward()
    {

    }

    // 反向传播
    bool _TestNNBackPropagation()
    {
        System.Random r = new System.Random(1);
        int m = 1; 
        NeuralNet p =
            new NeuralNet(r, new int[] { 1, 2, 1 }, ActivationFunction.Sigmoid);

        Matrix x = new Matrix(m, 1);
        Matrix y = new Matrix(m, 1);


        
        for (int i=0; i<m; i++)
        {
            double x_i = (double)(i+1);
            x.SetValue(i, 0, x_i / 4.0);
            y.SetValue(i, 0, x_i / 2.0);
        }
        

        Debug.Log("x:" + x);
        Debug.Log("y:" + y);

        Matrix[] A;

        Matrix h;
        for (int i=0; i<100; i++)
        {
            Debug.Log("=============="+i+"==============");
            h = p.ForwardPropagation(x, out A);
            p.BackPropagation(y, h, in A, 0.1);

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
