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

            Layers = [];

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        private void CreateInputLayer()
        {
            List<Neuron> inputNeurons = [];
            for (int i = 0; i < Topology.InputCount; i++)
            {
                Neuron neuron = new(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }

            Layer inputLayer = new(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }

        private void CreateHiddenLayers()
        {
            for (int j = 0; j < Topology?.HiddenLayers?.Count; j++)
            {
                List<Neuron> hiddenNeurons = [];
                Layer lastLayer = Layers.Last();

                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    Neuron neuron = new(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }

                Layer hiddenLayer = new(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateOutputLayer()
        {
            List<Neuron> outputNeurons = [];
            Layer lastLayer = Layers.Last();

            for (int i = 0; i < Topology.OutputCount; i++)
            {
                Neuron neuron = new(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }

            Layer outputLayer = new(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
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

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                List<double> signal = [inputSignals[i]];
                Neuron neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                Layer layer = Layers[i];
                List<double> previousLayerOutputs = Layers[i - 1].GetOutputs();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerOutputs);
                }
            }
        }

        public double Learn(double[] expected, double[,] dataset, int epoch, bool needNormalize)
        {
            if (needNormalize)
                dataset = Normalization(dataset);

            double error = 0.0;

            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    double[] input = GetRow(dataset, j);

                    error += BackPropagation(expected[j], input);
                }
            }

            return error / epoch;
        }

        private double BackPropagation(double expected, params double[] inputs)
        {
            double actual = FeedForward(inputs).Output;

            double difference = actual - expected;

            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                Layer layer = Layers[j];
                Layer prevLayer = Layers[j + 1];

                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    Neuron neuron = layer.Neurons[i];

                    for (int k = 0; k < prevLayer.NeuronCount; k++)
                    {
                        Neuron prevNeuron = prevLayer.Neurons[k];
                        double error = prevNeuron.Weights[i] * prevNeuron.Delta;

                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }

            return difference * difference;
        }

        private static double[,] Normalization(double[,] inputs)
        {
            double[,] result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int row = 0; row < inputs.GetLength(0); row++)
            {
                // Среднее значение сигнала
                double sum = 0.0;
                for (int item = 0; item < inputs.GetLength(1); item++)
                {
                    sum += inputs[row, item];
                }
                double average = sum / inputs.GetLength(1);

                // Стандартное квадратичное отклонение сигнала
                double error = 0.0;
                for (int item = 0; item < inputs.GetLength(1); item++)
                {
                    error += Math.Pow((inputs[row, item] - average), 2);
                }
                double stDev = Math.Sqrt(error / inputs.GetLength(1));

                // Нормализованные значения
                for (int item = 0; item < inputs.GetLength(1); item++)
                {
                    result[row, item] = (inputs[row, item] - average) / stDev;
                }
            }

            return result;
        }

        private static double[,] Scalling(double[,] inputs)
        {
            double[,] result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int row = 0; row < inputs.GetLength(0); row++)
            {
                double min = inputs[row, 0];
                double max = inputs[row, 0];

                for (int item = 1; item < inputs.GetLength(1); item++)
                {
                    double input = inputs[row, item];

                    if (input < min)
                    {
                        min = input;
                    }

                    if (input > max)
                    {
                        max = input;
                    }
                }

                double divider = max - min;

                for (int item = 1; item < inputs.GetLength(1); item++)
                {
                    result[row, item] = (inputs[row, item] - min) / divider;
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
