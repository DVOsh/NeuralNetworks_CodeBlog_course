using System.Data;

namespace NeuralNetwork_CodeBlog_course
{
    public class NeuralNetwork
    {
        public Topology Topology { get; }

        public List<Layer> Layers { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;

            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        public Neuron FeedForward(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }

            return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
        }

        public double Learn(double[] expected, double[,] dataset, int epoch, bool needNormalize)
        {
            if (needNormalize)
                dataset = Normalization(dataset);

            var error = 0.0;

            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var input = GetRow(dataset, j);

                    error += BackPropagation(expected[j], input);
                }
            }

            return error / epoch;
        }

        private double BackPropagation(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;

            var difference = actual - expected;

            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var prevLayer = Layers[j + 1];

                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];

                    for (int k = 0; k < prevLayer.NeuronCount; k++)
                    {
                        var prevNeuron = prevLayer.Neurons[k];
                        var error = prevNeuron.Weights[i] * prevNeuron.Delta;

                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }

            return difference * difference;
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerOutputs = Layers[i - 1].GetOutputs();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerOutputs);
                }
            }
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }

        }

        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }

            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }

        private void CreateHiddenLayers()
        {
            for (int j = 0; j < Topology?.HiddenLayers?.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();

                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }

                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();

            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }

            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        private static double[,] Scalling(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int row = 0; row < inputs.GetLength(0); row++)
            {
                var min = inputs[row, 0];
                var max = inputs[row, 0];

                for (int item = 1; item < inputs.GetLength(1); item++)
                {
                    var input = inputs[row, item];

                    if (input < min)
                    {
                        min = input;
                    }

                    if (input > max)
                    {
                        max = input;
                    }
                }

                var divider = max - min;

                for (int item = 1; item < inputs.GetLength(1); item++)
                {
                    result[row, item] = (inputs[row, item] - min) / divider;
                }
            }

            return result;
        }

        private static double[,] Normalization(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int row = 0; row < inputs.GetLength(0); row++)
            {
                // Среднее значение сигнала
                var sum = 0.0;
                for (int item = 0; item < inputs.GetLength(1); item++)
                {
                    sum += inputs[row, item];
                }
                var average = sum / inputs.GetLength(1);

                // Стандартное квадратичное отклонение сигнала
                var error = 0.0;
                for (int item = 0; item < inputs.GetLength(1); item++)
                {
                    error += Math.Pow((inputs[row, item] - average), 2);
                }
                var stDev = Math.Sqrt(error / inputs.GetLength(1));

                for (int item = 0; item < inputs.GetLength(1); item++)
                {
                    result[row, item] = (inputs[row, item] - average) / stDev;
                }
            }

            return result;
        }

        public static double[] GetRow(double[,] matrix, int row)
        {
            if (matrix.GetLength(0) <= row)
                throw new ArgumentException();

            double[] row_res = new double[matrix.GetLength(1)];
            for (int i = 0; i < matrix.GetLength(1); i++)
            {
                row_res[i] = matrix[row, i];
            }
            return row_res;
        }
    }
}
