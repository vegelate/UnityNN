using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System;

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
                Debug.Log("[" + i + "] " + tester.strName + "---------- [Pass]");
            }
            else
            {
                Debug.Log("[" + i + "] " + tester.strName + "---------- [Fail]");
            }
        }

        Debug.Log("测试完成，通过率 " + iPassCount + "/" + m_listTester.Count);
    }

    // 
    void _RegisterAllTester()
    {
        m_listTester.Add(new stTester("Matrix", _TestMatrix));
    }

    bool _TestMatrix()
    {

        return true;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
