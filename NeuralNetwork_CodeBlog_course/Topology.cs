using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork_CodeBlog_course
{
    public class Topology
    {
        public int InputCount { get; } //количество входов в нейронную сеть

        public int OutputCount { get; } 

        public double LearningRate { get; }

        public List<int>? HiddenLayers { get; }

        public Topology(int inputCount, int outputCount, double learningRate, params int[] layers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            LearningRate = learningRate;
            HiddenLayers = new List<int>();
            HiddenLayers?.AddRange(layers);
        }
    }
}
