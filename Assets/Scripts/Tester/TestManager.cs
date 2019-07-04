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
        m_listTester.Add(new stTester("NN Back", _TestNNBackPropagation));
    }

    bool _TestMatrix()
    {
        double[,] data1 = new double[5, 5];
        double[,] data2 = new double[5, 5];
        for (int x=0; x<5; x++)
        {
            for (int y=0; y<5; y++)
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

        Perceptron p = 
            new Perceptron(r, new int[] { 2, 4, 8 }, ActivationFunction.ReLU);

        Matrix input = Matrix.Random(1, 2, r);

        Matrix result = p.ForwardPropagation(input);

        Debug.Log("Result:" + result );

        return true;
    }

    // 反向传播
    bool _TestNNBackPropagation()
    {
        System.Random r = new System.Random(1);
        int m = 1; 
        Perceptron p =
            new Perceptron(r, new int[] { 2, 4, 6, 8 }, ActivationFunction.ReLU);

        Matrix x = Matrix.Random(m, 2, r);

        Matrix y = Matrix.Random(m, 8, r);

        Matrix h = p.ForwardPropagation(x);

        p.BackPropagation(y, h);

        return true;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
