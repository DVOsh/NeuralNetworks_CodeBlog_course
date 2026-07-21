using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork_CodeBlog_course
{
    public class Neuron
    {
        public List<double> Weights { get; }

        public NeuronType NeuronType { get; }

        public double Output { get; private set; }

        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Weights = new List<double>();

            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add(1);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            if (inputs.Count != Weights.Count)
                return 0;

            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            if (NeuronType != NeuronType.Input)
                Output = Sigmoid(sum);
            else
                Output = sum;

            return Output;
        }

        private static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -x));
        }

        public void SetWeights(params double[] weights)
        {
            // TODO: удалить после добавления возможности обучения сети.
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
