using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork_CodeBlog_course
{
    public class Neuron
    {
        public List<double> Weights { get; }

        public List<double> Inputs { get; }

        public NeuronType NeuronType { get; }

        public double Output { get; private set; }

        public double Delta { get; private set; }

        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Weights = new List<double>();
            Inputs = new List<double>();
            InitWeightsRandomValue(inputCount);
        }

        private void InitWeightsRandomValue(int inputCount)
        {
            var rnd = new Random();

            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add(NeuronType == NeuronType.Input ? 1 : rnd.NextDouble());
                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            if (inputs.Count != Weights.Count)
                throw new ArgumentException("Inputs count must be must be equal to weights count!");

            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

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

        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid * (1 - sigmoid);
            return result;
        }

        public void SetWeights(params double[] weights)
        {
            // TODO: удалить после добавления возможности обучения сети.
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }

        public void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Input)
                return;

            Delta = error * SigmoidDx(this.Output);

            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeight = weight - input * Delta * learningRate;
                Weights[i] = newWeight;
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
