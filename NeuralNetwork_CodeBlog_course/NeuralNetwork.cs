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
            for (int i = 0; i < Topology.InputsCount; i++)
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

            for (int i = 0; i < Topology.OutputsCount; i++)
            {
                Neuron neuron = new(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }

            Layer outputLayer = new(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        public double FeedForward(double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);

            for (int i = 1; i < Layers.Count; i++)
            {
                Layer layer = Layers[i];
                double[] previousLayerOutputs = Layers[i - 1].GetOutputs();

                foreach (Neuron neuron in layer.Neurons)
                {
                    neuron.CalcNeuronOutput(previousLayerOutputs);
                }
            }

            if (Topology.OutputsCount == 1)
            {
                return Layers.Last().Neurons[0].Output;
            }

            return Layers.Last().Neurons.OrderByDescending(n => n.Output).First().Output;
        }

        private void SendSignalsToInputNeurons(double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                double signal = inputSignals[i];
                Neuron neuron = Layers[0].Neurons[i];

                neuron.CalcNeuronOutput([signal]);
            }
        }

        public double Learn(Dataset dataset, int epoch, bool needNormalize, double testPct)
        {
            double[,] inputs = GetInputSignalsMatrixFromDataset(dataset);

            if (needNormalize)
                inputs = Normalization(inputs);

            double error = 0.0;

            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < dataset.Results.Count; j++)
                {
                    double[] input = GetRow(inputs, j);

                    error += BackPropagation(dataset.Results[j], input);
                }
            }

            return error / epoch;
        }

        private double BackPropagation(double expected, double[] inputs)
        {
            // Начальный результат нейросети со случайными весами
            double actual = FeedForward(inputs);

            // Разница между начальным результатом и ожидаемым значением
            double difference = actual - expected;

            // Корректировка весов в выходном слое
            foreach (Neuron neuron in Layers.Last().Neurons)
            {
                neuron.WeightsCorrection(difference, Topology.LearningRate);
            }

            // Перебор слоев в обратном порядке, за исключением выходного слоя
            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                Layer layer = Layers[j];
                Layer nextLayer = Layers[j + 1];

                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    Neuron neuron = layer.Neurons[i];

                    for (int k = 0; k < nextLayer.NeuronCount; k++)
                    {
                        Neuron nextLayerNeuron = nextLayer.Neurons[k];
                        double error = nextLayerNeuron.Weights[i] * nextLayerNeuron.Delta;

                        neuron.WeightsCorrection(error, Topology.LearningRate);
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

        private static double[,] GetInputSignalsMatrixFromDataset(Dataset dataset)
        {
            double[,] inputSignals = new double[dataset.Inputs.Count, dataset.Inputs[0].Length];
            for (int i = 0; i < inputSignals.GetLength(0); i++)
            {
                for (var j = 0; j < inputSignals.GetLength(1); j++)
                {
                    inputSignals[i, j] = dataset.Inputs[i][j];
                }
            }

            return inputSignals;
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
