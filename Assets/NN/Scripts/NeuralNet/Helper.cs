using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NN
{
    public static class Helper
    {
        public static void PrintDim(string name, Matrix m)
        {
            Debug.Log(name + "X:" + m.X + ", Y:" + m.Y);
        }
    }
}

