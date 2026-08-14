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
                string[] datasetHeaders = ["temp", "age", "smoking", "food"];
                double[,] datasetInputs = new double[,]
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
                double[] datasetResults = [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1];
                Dataset dataset = new(datasetHeaders, datasetInputs, datasetResults, 20);

                Topology topology = new(4, 1, 0.1, 2);
                NeuralNetwork neuralNetwork = new(topology);
                neuralNetwork.Learn(dataset, 100, true);

                List<double> testResults = neuralNetwork.TestRestData(dataset);
                for (int i = 0; i < testResults.Count; i++)
                {
                    Assert.AreEqual(dataset.Results[dataset.LearnCount + i], Math.Round(testResults[i]), $"row: {i}");
                }

            }

            [TestMethod]
            public void Heart_DatasetTest()
            {
                Dataset dataset = new("../../../../../Datasets/heart_decrease/heart.csv", 20);
                dataset.NormalizeInputs();

                Topology topology = new(dataset.Inputs[0].Length, 1, 0.1, dataset.Inputs[0].Length / 2);
                NeuralNetwork neuralNetwork = new(topology);
                neuralNetwork.Learn(dataset, 10000, true);

                List<double> testResults = neuralNetwork.TestRestData(dataset);
                for (int i = 0; i < testResults.Count; i++)
                {
                    Assert.AreEqual(dataset.Results[dataset.LearnCount + i], Math.Round(testResults[i]), $"row: {i}");
                }
            }

            //[TestMethod]
            //public void RecognizeImages()
            //{
            //    var parasitizedPath = @"D:\Coding\CSharp\Education\CodeBlog\Datasets\cell_images\Parasitized";
            //    var uninfectedPath = @"D:\Coding\CSharp\Education\CodeBlog\Datasets\cell_images\Uninfected";

            //    var converter = new PictureConverter();
            //    var testParasitedImageInput = converter.Convert(@"D:\Coding\CSharp\Education\CodeBlog\NeuralNetwork_CodeBlog_course\NeuralNetwork_Tests\Images\Parasitized.png");
            //    var testUninfectedImageInput = converter.Convert(@"D:\Coding\CSharp\Education\CodeBlog\NeuralNetwork_CodeBlog_course\NeuralNetwork_Tests\Images\Uninfected.png");


            //    var topology = new Topology(testParasitedImageInput.Length, 1, 0.1, testParasitedImageInput.Length / 2);
            //    var neuralNetwork = new NeuralNetwork(topology);

            //    double[,] parasitizedInputs = GetData(parasitizedPath, converter, testParasitedImageInput);
            //    neuralNetwork.Learn([1.0], parasitizedInputs, 10000, false);

            //    double[,] uninfectedInputs = GetData(uninfectedPath, converter, testUninfectedImageInput);
            //    neuralNetwork.Learn([0.0], uninfectedInputs, 10000, false);

            //    var par = neuralNetwork.FeedForward(testParasitedImageInput.Select(t => (double)t).ToArray());
            //    var uninf = neuralNetwork.FeedForward(testUninfectedImageInput.Select(t => (double)t).ToArray());

            //    Assert.AreEqual(1, Math.Round(par, 2));
            //    Assert.AreEqual(0, Math.Round(uninf, 2));
            //}

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
        }
    }
}
