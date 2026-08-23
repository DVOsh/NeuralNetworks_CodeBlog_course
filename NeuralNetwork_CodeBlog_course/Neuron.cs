namespace NeuralNetwork_CodeBlog_course
{
    public class Neuron
    {
        public List<double> Weights { get; }

        public List<double> Signals { get; }

        public NeuronType NeuronType { get; }

        public Func<double, double>? AcFunc { get; set; }

        public Func<double, double>? AcFuncDx { get; set; }

        public double Output { get; private set; }

        public double Delta { get; private set; }

        public Neuron(int inputCount, NeuronType type = NeuronType.Hidden)
        {
            NeuronType = type;
            Weights = [];
            Signals = [];
            NeuronStartInit(inputCount);
        }

        public void SetWeights(params double[] weights)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }

        private void NeuronStartInit(int inputCount)
        {
            Random rnd = new();

            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add(NeuronType == NeuronType.Input ? 1 : rnd.NextDouble());
                Signals.Add(0);
            }
        }

        public double CalcNeuronOutput(double[] signals)
        {
            if (signals.Length != Weights.Count)
                throw new ArgumentException("Inputs count must be must be equal to weights count!");

            double sum = 0.0;


            for (int i = 0; i < signals.Length; i++)
            {
                Signals[i] = signals[i];
                sum += signals[i] * Weights[i];
            }

            if (NeuronType != NeuronType.Input)
            {
                Output = AcFunc != null ? AcFunc(sum) : sum;
            }
            else
            {
                Output = sum;
            }

            return Output;
        }

        public void WeightsCorrection(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Input)
                return;

            Delta = AcFuncDx != null ? AcFuncDx(Output) * error
                                     : Output * error;

            for (int i = 0; i < Weights.Count; i++)
            {
                double weight = Weights[i];
                double signal = Signals[i];

                double newWeight = weight - signal * Delta * learningRate;
                Weights[i] = newWeight;
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
