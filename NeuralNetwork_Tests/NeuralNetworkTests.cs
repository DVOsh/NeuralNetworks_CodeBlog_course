using NeuralNetwork_CodeBlog_course;

namespace NeuralNetwork_Tests
{
    [TestClass]
    public sealed class NeuralNetworkTests
    {
        [TestClass]
        public sealed class NeuralNetorkTests
        {
            [TestMethod]
            public void FeedForwardTest()
            {
                var topology = new Topology(4, 1, 0.1, 2);
                var neuralNetwork = new NeuralNetwork(topology);
                neuralNetwork.Layers[1].Neurons[0].SetWeights(0.5, -0.1, 0.3, -0.1);
                neuralNetwork.Layers[1].Neurons[1].SetWeights(0.1, -0.3, 0.7, -0.3);
                neuralNetwork.Layers[2].Neurons[0].SetWeights(1.2, 0.8);

                var result = neuralNetwork.FeedForward([1, 0, 0, 0]);

                Assert.AreEqual(1, Math.Round(result));
            }

            [TestMethod]
            public void BackPropagation_Learn_Test()
            {
                var expecteds = new double[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
                var dataset = new double[,]
                {
                    // Результат - Пациент болен - 1
                    //             Пациент здоров - 0
            
                    // Неправильная температура T
                    // Хороший возраст A
                    // Курит S
                    // Правильно питается F
                    //T  A  S  F
                    { 0, 0, 0, 0 },
                    { 0, 0, 0, 1 },
                    { 0, 0, 1, 0 },
                    { 0, 0, 1, 1 },
                    { 0, 1, 0, 0 },
                    { 0, 1, 0, 1 },
                    { 0, 1, 1, 0 },
                    { 0, 1, 1, 1 },
                    { 1, 0, 0, 0 },
                    { 1, 0, 0, 1 },
                    { 1, 0, 1, 0 },
                    { 1, 0, 1, 1 },
                    { 1, 1, 0, 0 },
                    { 1, 1, 0, 1 },
                    { 1, 1, 1, 0 },
                    { 1, 1, 1, 1 }
                };

                var topology = new Topology(4, 1, 0.1, 2);
                var neuralNetwork = new NeuralNetwork(topology);
                var difference = neuralNetwork.Learn(expecteds, dataset, 10000, false);

                var results = new List<double>();
                //foreach (var data in dataset)
                //{
                //    results.Add(neuralNetwork.FeedForward(data).Output);
                //}

                for (int i = 0; i < dataset.GetLength(0); i++)
                {
                    double[] row = NeuralNetwork.GetRow(dataset, i);
                    results.Add(neuralNetwork.FeedForward(row));
                }

                for (int i = 0; i < results.Count; i++)
                {
                    var expected = Math.Round(expecteds[i], 3);
                    var actual = Math.Round(results[i], 3);
                    Assert.AreEqual(expected, actual);
                }
            }

            [TestMethod]
            public void Heart_DatasetTest()
            {
                var outputs = new List<double>();
                var inputs = new List<double[]>();

                using var sr = new StreamReader("../../../../../Datasets/heart_decrease/heart.csv");
                var header = sr.ReadLine();

                while (!sr.EndOfStream)
                {
                    var row = sr.ReadLine();
                    //var values = row.Split(',').Select(v => Convert.ToDouble(v)).ToList();
                    var values = row.Split(',').Select(v => double.Parse(v.Replace(".", ","))).ToList();
                    var output = values.Last();
                    var input = values.Take(values.Count - 1).ToArray();

                    outputs.Add(output);
                    inputs.Add(input);
                }

                var inputSignals = new double[inputs.Count, inputs[0].Length];
                for (int i = 0; i < inputSignals.GetLength(0); i++)
                {
                    for (var j = 0; j < inputSignals.GetLength(1); j++)
                    {
                        inputSignals[i, j] = inputs[i][j];
                    }
                }

                var topology = new Topology(inputs[0].Length, 1, 0.1, inputs[0].Length / 2);
                var neuralNetwork = new NeuralNetwork(topology);
                var difference = neuralNetwork.Learn(outputs.ToArray(), inputSignals, 10000, false);

                var results = new List<double>();
                for (int i = 0; i < outputs.Count; i++)
                {
                    results.Add(neuralNetwork.FeedForward(inputs[i]));
                }

                for (int i = 0; i < results.Count; i++)
                {
                    var expected = Math.Round(outputs[i], 3);
                    var actual = Math.Round(results[i], 3);
                    Assert.AreEqual(expected, actual);
                }
            }

            [TestMethod]
            public void RecognizeImages()
            {
                var parasitizedPath = @"D:\Coding\CSharp\Education\CodeBlog\Datasets\cell_images\Parasitized";
                var uninfectedPath = @"D:\Coding\CSharp\Education\CodeBlog\Datasets\cell_images\Uninfected";

                var converter = new PictureConverter();
                var testParasitedImageInput = converter.Convert(@"D:\Coding\CSharp\Education\CodeBlog\NeuralNetwork_CodeBlog_course\NeuralNetwork_Tests\Images\Parasitized.png");
                var testUninfectedImageInput = converter.Convert(@"D:\Coding\CSharp\Education\CodeBlog\NeuralNetwork_CodeBlog_course\NeuralNetwork_Tests\Images\Uninfected.png");


                var topology = new Topology(testParasitedImageInput.Length, 1, 0.1, testParasitedImageInput.Length / 2);
                var neuralNetwork = new NeuralNetwork(topology);

                double[,] parasitizedInputs = GetData(parasitizedPath, converter, testParasitedImageInput);
                neuralNetwork.Learn([1.0], parasitizedInputs, 10000, false);

                double[,] uninfectedInputs = GetData(uninfectedPath, converter, testUninfectedImageInput);
                neuralNetwork.Learn([0.0], uninfectedInputs, 10000, false);

                var par = neuralNetwork.FeedForward(testParasitedImageInput.Select(t => (double)t).ToArray());
                var uninf = neuralNetwork.FeedForward(testUninfectedImageInput.Select(t => (double)t).ToArray());

                Assert.AreEqual(1, Math.Round(par, 2));
                Assert.AreEqual(0, Math.Round(uninf, 2));
            }

            private static double[,] GetData(string parasitizedPath, PictureConverter converter, double[] testImageInput)
            {
                var parasitizedImages = Directory.GetFiles(parasitizedPath);
                var datasetSize = 100;
                var parasitizedInputs = new double[datasetSize, testImageInput.Length];
                for (int i = 0; i < datasetSize; i++)
                {
                    var convertedImageInput = converter.Convert(parasitizedImages[i]);

                    for (int j = 0; j < convertedImageInput.Length; j++)
                    {
                        parasitizedInputs[i, j] = convertedImageInput[j];
                    }
                }

                return parasitizedInputs;
            }

            [TestMethod]
            public void DimensionTest()
            {
                var dataset = new double[,]
                {
                    // Результат - Пациент болен - 1
                    //             Пациент здоров - 0
            
                    // Неправильная температура T
                    // Хороший возраст A
                    // Курит S
                    // Правильно питается F
                    //T  A  S  F
                    { 0, 0, 0, 0 },
                    { 0, 0, 0, 1 },
                    { 0, 0, 1, 0 },
                    { 0, 0, 1, 1 },
                    { 0, 1, 0, 0 },
                    { 0, 1, 0, 1 },
                    { 0, 1, 1, 0 },
                    { 0, 1, 1, 1 },
                    { 1, 0, 0, 0 },
                    { 1, 0, 0, 1 },
                    { 1, 0, 1, 0 },
                    { 1, 0, 1, 1 },
                    { 1, 1, 0, 0 },
                    { 1, 1, 0, 1 },
                    { 1, 1, 1, 0 },
                    { 1, 1, 1, 1 },
                };

                Console.WriteLine(dataset.GetLength(0));
                Console.WriteLine(dataset.GetLength(1));
            }
        }
    }
}
