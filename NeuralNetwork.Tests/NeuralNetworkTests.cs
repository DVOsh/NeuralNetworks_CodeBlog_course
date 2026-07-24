using NeuralNetwork_CodeBlog_course;

namespace NeuralNetwork.Tests
{
    public class NeuralNetworkTests
    {
        [Fact]
        public void FeedForwardTest()
        {
            var topology = new Topology(4, 1, 0.1, 2);
            var neuralNetwork = new NeuralNetwork_CodeBlog_course.NeuralNetwork(topology);
            neuralNetwork.Layers[1].Neurons[0].SetWeights(0.5, -0.1, 0.3, -0.1);
            neuralNetwork.Layers[1].Neurons[1].SetWeights(0.1, -0.3, 0.7, -0.3);
            neuralNetwork.Layers[2].Neurons[0].SetWeights(1.2, 0.8);

            var result = neuralNetwork.FeedForward([1, 0, 0, 0]).Output;

            Assert.Equal(1, Math.Round(result));
        }

        [Fact]
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
                { 1, 1, 1, 1 },
            };

            var topology = new Topology(4, 1, 0.1, 5);
            var neuralNetwork = new NeuralNetwork_CodeBlog_course.NeuralNetwork(topology);
            var difference = neuralNetwork.Learn(expecteds, dataset, 1000000);

            var results = new List<double>();
            //foreach (var data in dataset)
            //{
            //    results.Add(neuralNetwork.FeedForward(data).Output);
            //}

            for (int i = 0; i < dataset.GetLength(0); i++)
            {
                double[] row = NeuralNetwork_CodeBlog_course.NeuralNetwork.GetRow(dataset, i);
                results.Add(neuralNetwork.FeedForward(row).Output);
            }

            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(expecteds[i], 3);
                var actual = Math.Round(results[i], 3);
                Assert.Equal(expected, actual);
            }
        }
    }
}
