using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork_CodeBlog_course
{
    public static class ActivationFunctions
    {
        public static (Func<double, double>, Func<double, double>) HiddenLayerFunctions { get; set; }

        public static (Func<double, double>, Func<double, double>) OutputLayerFunctions { get; set; }

        public static void SetHiddenLayerFunctions(FunctionsType type)
        {
            HiddenLayerFunctions = type switch
            {
                FunctionsType.Sigmoid => (Sigmoid, SigmoidDx),
                _ => throw new ArgumentOutOfRangeException(nameof(type), $"Not expected activation function type value: {type}")
            };
        }

        public static void SetOutputLayerFunctions(FunctionsType type)
        {
            OutputLayerFunctions = type switch
            {
                FunctionsType.Sigmoid => (Sigmoid, SigmoidDx),
                _ => throw new ArgumentOutOfRangeException(nameof(type), $"Not expected activation function type value: {type}")

            };
        }

        public static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -x));
        }

        private static double SigmoidDx(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }
    }
}
