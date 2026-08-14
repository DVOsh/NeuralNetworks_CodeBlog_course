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

        public void Learn(Dataset dataset, int epoch, bool needShuffle = false)
        {
            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < dataset.LearnCount; j++) 
                {
                    int index = dataset.Indexes[j];
                    double[] inputs = dataset.Inputs[index];         

                    Backpropagation(dataset.Results[index], inputs);
                }

                if (needShuffle)
                    dataset.ShuffleLearnDataIndexes();
            }
        }

        private double Backpropagation(double expected, double[] inputs)
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

        // Функия для тестирования нейросети на значениях, оставшихся после обучения
        public List<double> TestRestData(Dataset dataset)
        {
            List<double> results = [];

            for (int i = dataset.LearnCount; i < dataset.Inputs.Count; i++)
            {
                results.Add(FeedForward(dataset.Inputs[i]));
            }

            return results;
        }
    }
}
