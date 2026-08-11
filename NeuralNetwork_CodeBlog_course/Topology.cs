namespace NeuralNetwork_CodeBlog_course
{
    public class Topology
    {
        public int InputsCount { get; } //количество входов в нейронную сеть

        public int OutputsCount { get; } 

        public double LearningRate { get; }

        public List<int>? HiddenLayers { get; }

        public Topology(int inputCount, int outputCount, double learningRate, params int[] neuronsCountInHiddenLayers)
        {
            InputsCount = inputCount;
            OutputsCount = outputCount;
            LearningRate = learningRate;
            HiddenLayers = [];
            HiddenLayers?.AddRange(neuronsCountInHiddenLayers);
        }
    }
}
