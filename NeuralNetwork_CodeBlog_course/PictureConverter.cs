using System.Drawing;

namespace NeuralNetwork_CodeBlog_course
{
    public class PictureConverter
    {
        public int Width { get; set; }

        public int Height { get; set; }

        public int Boundary { get; set; } = 128; //hardcode

        public double[] Convert(string path)
        {
            List<double> result = [];

            Bitmap image = new(path);
            Bitmap resizeImage = new(image, new Size(50, 50)); //hardcode
            Height = resizeImage.Height;
            Width = resizeImage.Width;

            for (int y = 0; y < resizeImage.Height; y++)
            {
                for (int x = 0; x < resizeImage.Width; x++)
                {
                    Color pixel = resizeImage.GetPixel(x, y);
                    double value = Brightness(pixel);
                    result.Add(value);
                }
            }

            return [.. result];
        }

        private int Brightness(Color pixel)
        {
            double result = 0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B;
            return result < Boundary ? 0 : 1;
        }

        public void Save(string path, double[] pixels)
        {
            Bitmap image = new(Width, Height);
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    Color color = pixels[x + y * Width] == 1 ? Color.White : Color.Black;
                    image.SetPixel(x, y, color);
                }
            }

            image.Save(path);
        }
    }
}
