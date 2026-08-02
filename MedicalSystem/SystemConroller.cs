using NeuralNetwork_CodeBlog_course;
using System;
using System.Collections.Generic;
using System.Text;

namespace MedicalSystem
{
    public class SystemConroller
    {
        public NeuralNetwork DataNetwork { get; }

        public NeuralNetwork ImageNetwork { get; }

        public SystemConroller()
        {
            var dataTopology = new Topology(14, 1, 0.1, 7);
            DataNetwork = new NeuralNetwork(dataTopology);

            var imageTopology = new Topology(400, 1, 0.1, 200);
            ImageNetwork = new NeuralNetwork(imageTopology);
        }
    }
}
