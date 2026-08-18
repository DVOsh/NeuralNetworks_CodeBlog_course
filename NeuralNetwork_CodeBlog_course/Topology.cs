namespace NeuralNetwork_CodeBlog_course
{
    public class Topology
    {
        public int InputsCount { get; } //количество входов в нейронную сеть

        public int OutputsCount { get; } 

        public List<int>? HiddenLayers { get; }

        public Topology(int inputCount, int outputCount, params int[] neuronsCountInHiddenLayers)
        {
            InputsCount = inputCount;
            OutputsCount = outputCount;
            HiddenLayers = [];
            HiddenLayers?.AddRange(neuronsCountInHiddenLayers);
        }
    }
}
