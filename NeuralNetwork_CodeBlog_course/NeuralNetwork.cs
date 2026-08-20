using System.Data;

namespace NeuralNetwork_CodeBlog_course
{
    public class NeuralNetwork
    {
        public List<Layer> Layers { get; }

        public double LearningRate { get; }

        public NeuralNetwork(Topology topology, double learningRate, FunctionsType hiddenLayerType, FunctionsType outputLayerType)
        {
            Layers = [];
            LearningRate = learningRate;
            ActivationFunctions.SetHiddenLayerFunctions(hiddenLayerType);
            ActivationFunctions.SetOutputLayerFunctions(outputLayerType);

            CreateLayers(topology);
        }

        private void CreateLayers(Topology topology)
        {
            Layers.Add(new Layer(NeuronType.Input, topology.InputsCount));
            for (int i = 0; i < topology.HiddenLayers.Count; i++)
            {
                Layers.Add(new Layer(NeuronType.Hidden, topology.HiddenLayers[i]));
            }
            Layers.Add(new Layer(NeuronType.Output, topology.OutputsCount));
        }

        public double FeedForward(double[] inputSignals, bool needNormalize = false)
        {
            if (needNormalize)
                inputSignals = Dataset.NormalizeInputs(inputSignals);

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

            if (Layers.Last().NeuronsCount == 1) // проверить значение
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
                    double[] inputs = dataset.NeedNormalize
                                      ? Dataset.NormalizeInputs(dataset.Inputs[index])
                                      : dataset.Inputs[index];

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
                neuron.WeightsCorrection(difference, LearningRate);
            }

            // Перебор слоев в обратном порядке, за исключением выходного слоя
            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                Layer layer = Layers[j];
                Layer nextLayer = Layers[j + 1];

                for (int i = 0; i < layer.NeuronsCount; i++)
                {
                    Neuron neuron = layer.Neurons[i];

                    for (int k = 0; k < nextLayer.NeuronsCount; k++)
                    {
                        Neuron nextLayerNeuron = nextLayer.Neurons[k];
                        double error = nextLayerNeuron.Weights[i] * nextLayerNeuron.Delta;

                        neuron.WeightsCorrection(error, LearningRate);
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
