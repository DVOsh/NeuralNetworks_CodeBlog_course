namespace NeuralNetwork_CodeBlog_course
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }

        public int NeuronCount => Neurons?.Count ?? 0;

        public NeuronType Type { get; }

        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Hidden)
        {
            // TODO: проверить все входные нейроны на соответствие типу

            Neurons = neurons;
            Type = type;
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
