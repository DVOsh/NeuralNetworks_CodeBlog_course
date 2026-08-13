namespace NeuralNetwork_CodeBlog_course
{
    public class Dataset
    {
        private double testPct;

        public List<string> Headers { get; private set; } = [];

        public List<double[]> Inputs { get; private set; } = [];

        public List<double> Results { get; private set; } = [];

        public int LearnCount { get; private set; }
        public int TestCount { get; private set; }

        // Задание процента выборки для проведения тестов обучения
        public double TestPct
        {
            get => testPct;
            init
            {
                if (value < 0)
                    throw new InvalidOperationException("Test percent must be greater than 0!");

                if (value > 90)
                    throw new InvalidOperationException("The neural network must learn from something!");

                testPct = value;

                TestCount = (int)(Inputs.Count * testPct / 100);
                LearnCount = Inputs.Count - TestCount;
            }
        }

        public Dataset(string[] headers, double[,] inputs, double[] results, double testPct = 0)
        {
            if (headers == null)
                throw new ArgumentNullException("Set headers!");
            if (inputs.GetLength(0) != results.Length)
                throw new ArgumentException("Rows count must be equals to results count");

            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                Inputs.Add(GetRow(inputs, i));
            }

            Results = results.ToList();
            TestPct = testPct;
        }

        public Dataset(string path, double testPct)
        {
            using StreamReader sr = new(path);
            Headers = sr.ReadLine()?.Split(',').SkipLast(1).ToList()
                ?? throw new ArgumentException("Path isn't correct or file is empty!");

            while (!sr.EndOfStream)
            {
                string? row = sr.ReadLine()
                    ?? throw new Exception("File doesn't contain rows");
                List<double> values = row.Split(',')
                                         .Select(v => double.Parse(v.Replace(".", ",")))
                                         .ToList();
                double result = values.Last();
                double[] input = values.Take(values.Count - 1).ToArray();

                Results.Add(result);
                Inputs.Add(input);
            }

            TestPct = testPct;
        }

        public void NormalizeInputs()
        {
            if (Inputs.Count < 1)
                throw new InvalidOperationException("Dataset is empty!");

            int rowsCount = Inputs.Count;
            int headersCount = Headers.Count;
            double[,] result = new double[rowsCount, headersCount];

            for (int row = 0; row < rowsCount; row++)
            {
                // Среднее значение сигнала
                double sum = 0.0;
                for (int item = 0; item < headersCount; item++)
                {
                    sum += Inputs[row][item];
                }
                double average = sum / headersCount;

                // Стандартное квадратичное отклонение сигнала
                double error = 0.0;
                for (int item = 0; item < headersCount; item++)
                {
                    error += Math.Pow((Inputs[row][item] - average), 2);
                }
                double stDev = Math.Sqrt(error / headersCount);

                // Нормализованные значения
                for (int item = 0; item < headersCount; item++)
                {
                    result[row, item] = (Inputs[row][item] - average) / stDev;
                }
            }
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

        private static double[] GetRow(double[,] matrix, int row)
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
