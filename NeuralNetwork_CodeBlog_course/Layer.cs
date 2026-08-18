namespace NeuralNetwork_CodeBlog_course
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }

        public int NeuronsCount => Neurons?.Count ?? 0;

        public static int PrevLayerNeuronsCount { get; private set; }

        public NeuronType Type { get; }

        public Layer(NeuronType type, int neuronsCount)
        {
            Type = type;
            Neurons = type switch
            {
                NeuronType.Input => CreateInputNeurons(neuronsCount),
                NeuronType.Hidden => CreateHiddenNeurons(neuronsCount, PrevLayerNeuronsCount),
                NeuronType.Output => CreateOutputNeurons(neuronsCount, PrevLayerNeuronsCount),
                _ => throw new ArgumentOutOfRangeException(nameof(type), $"Not expected neuron type value: {type}")
            };
            PrevLayerNeuronsCount = NeuronsCount;
        }

        private static List<Neuron> CreateInputNeurons(int neuronsCount)
        {
            List<Neuron> inputNeurons = [];
            for (int i = 0; i < neuronsCount; i++)
            {
                Neuron neuron = new(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }

            return inputNeurons;
        }

        private static List<Neuron> CreateHiddenNeurons(int neuronsCount, int neuronInputsCount)
        {
            List<Neuron> hiddenNeurons = [];

            for (int i = 0; i < neuronsCount; i++)
            {
                Neuron neuron = new(neuronInputsCount); // !!! inputsCount???
                (neuron.AcFunc, neuron.AcFuncDx) = ActivationFunctions.HiddenLayerFunctions;
                hiddenNeurons.Add(neuron);
            }

            return hiddenNeurons;
        }

        private static List<Neuron> CreateOutputNeurons(int neuronsCount, int neuronInputsCount)
        {
            List<Neuron> outputNeurons = [];

            for (int i = 0; i < neuronsCount; i++)
            {
                Neuron neuron = new(neuronInputsCount, NeuronType.Output);
                (neuron.AcFunc, neuron.AcFuncDx) = ActivationFunctions.OutputLayerFunctions;
                outputNeurons.Add(neuron);
            }

            return outputNeurons;
        }

        public double[] GetOutputs()
        {
            double[] result = new double[Neurons.Count];

            for (int i = 0; i < Neurons.Count; i++)
            {
                result[i] = Neurons[i].Output;
            }

            return result;
        }

        public override string ToString()
        {
            return Type.ToString();
        }
    }
}
