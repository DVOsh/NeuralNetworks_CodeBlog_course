using System.Data;
using System.Runtime.ExceptionServices;

namespace NeuralNetwork_CodeBlog_course
{
    public class NeuralNetwork
    {
        public List<Layer> Layers { get; }

        public double LearningRate { get; }

        public NeuralNetwork(Topology topology, double learningRate, FunctionsType? hiddenLayerType, FunctionsType? outputLayerType)
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
            if (topology.HiddenLayers != null)
            {
                for (int i = 0; i < topology.HiddenLayers.Count; i++)
                {
                    Layers.Add(new Layer(NeuronType.Hidden, topology.HiddenLayers[i]));
                }
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

            if (Layers.Last().NeuronsCount == 1)
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
            Console.WriteLine("Start learning a neural network:");
            Console.WriteLine();

            for (int i = 0; i < epoch; i++)
            {
                Console.WriteLine("===================================================");
                Console.WriteLine($"                     Epoch {i + 1}");
                Console.WriteLine("===================================================");
                Console.WriteLine();

                for (int j = 0; j < dataset.LearnCount; j++) 
                {
                    int index = dataset.Indexes[j];
                    double[] inputs = dataset.NeedNormalize
                                      ? Dataset.NormalizeInputs(dataset.Inputs[index])
                                      : dataset.Inputs[index];

                    Backpropagation(dataset.Results[index], inputs);
                }

                NeuronDataLog();

                if (needShuffle)
                    dataset.ShuffleLearnDataIndexes();
            }
        }

        private double Backpropagation(double expected, double[] inputs)
        {
            // Начальный результат нейросети [со случайными весами]
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
                results.Add(FeedForward(dataset.Inputs[i], dataset.NeedNormalize));
            }

            return results;
        }

        private void NeuronDataLog()
        {
            Console.WriteLine("Input layer:");
            Console.WriteLine("---------------------------------------------------");
            Console.WriteLine("{0,6}|{1,10}|{2,10}|{3,10}|{4,10}", "Neuron", "Inputs", "Weights", "Delta", "Output");
            Console.WriteLine("---------------------------------------------------");
            for (int i = 0; i < Layers[0].Neurons.Count; i++)
            {
                Neuron neuron = Layers[0].Neurons[i];
                for (int j = 0; j < neuron.Signals.Count; j++)
                {
                    double inputSignal = Math.Round(neuron.Signals[j], 5);
                    double weight = Math.Round(neuron.Weights[j], 5);
                    double delta = Math.Round(neuron.Delta, 5);
                    double output = Math.Round(neuron.Output, 5);
                    if (j == 0)
                    {
                        Console.WriteLine("{0,6}|{1,10}|{2,10}|{3,10}|{4,10}", i, inputSignal, weight, delta, output);
                    }
                    else
                    {
                        Console.WriteLine("{0,6}|{1,10}|{2,10}|{3,10}|{4,10}", "", inputSignal, weight, "", "");
                    }
                }
                Console.WriteLine("---------------------------------------------------");
            }
            Console.WriteLine();


            for (int i = 1; i < Layers.Count - 1; i ++)
            {
                Console.WriteLine($"Hidden layer {i}:");
                Console.WriteLine("---------------------------------------------------");
                Console.WriteLine("{0,6}|{1,10}|{2,10}|{3,10}|{4,10}", "Neuron", "Inputs", "Weights", "Delta", "Output");
                Console.WriteLine("---------------------------------------------------");
                for (int k = 0; k < Layers[i].Neurons.Count; k++)
                {
                    Neuron neuron = Layers[i].Neurons[k];
                    for (int j = 0; j < neuron.Signals.Count; j++)
                    {
                        double inputSignal = Math.Round(neuron.Signals[j], 5);
                        double weight = Math.Round(neuron.Weights[j], 5);
                        double delta = Math.Round(neuron.Delta, 5);
                        double output = Math.Round(neuron.Output, 5);
                        if (j == 0)
                        {
                            Console.WriteLine("{0,6}|{1,10}|{2,10}|{3,10}|{4,10}", k, inputSignal, weight, delta, output);
                        }
                        else
                        {
                            Console.WriteLine("{0,6}|{1,10}|{2,10}|{3,10}|{4,10}", "", inputSignal, weight, "", "");
                        }
                    }
                    Console.WriteLine("---------------------------------------------------");
                }
                Console.WriteLine();
            }


            Console.WriteLine("Output layer:");
            Console.WriteLine("---------------------------------------------------");
            Console.WriteLine("{0,6}|{1,10}|{2,10}|{3,10}|{4,10}", "Neuron", "Inputs", "Weights", "Delta", "Output");
            Console.WriteLine("---------------------------------------------------");
            for (int i = 0; i < Layers.Last().Neurons.Count; i++)
            {
                Neuron neuron = Layers.Last().Neurons[i];
                for (int j = 0; j < neuron.Signals.Count; j++)
                {
                    double inputSignal = Math.Round(neuron.Signals[j], 5);
                    double weight = Math.Round(neuron.Weights[j], 5);
                    double delta = Math.Round(neuron.Delta, 5);
                    double output = Math.Round(neuron.Output, 5);
                    if (j == 0)
                    {
                        Console.WriteLine("{0,6}|{1,10}|{2,10}|{3,10}|{4,10}", i, inputSignal, weight, delta, output);
                    }
                    else
                    {
                        Console.WriteLine("{0,6}|{1,10}|{2,10}|{3,10}|{4,10}", "", inputSignal, weight, "", "");
                    }
                }
                Console.WriteLine("---------------------------------------------------");
            }
            double nnoutput = Math.Round(Layers.Last().Neurons[0].Output, 10);
            Console.WriteLine($"####### NEURAL NETWORK OUTPUT: {nnoutput} #######");
            Console.WriteLine("===================================================");
            Console.WriteLine();
            Console.WriteLine();
        }
    }
}
