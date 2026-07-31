using System.Drawing;

namespace NeuralNetwork_CodeBlog_course
{
    public class PictureConverter
    {
        public int Width { get; set; }

        public int Height { get; set; }

        public int Boundary { get; set; } = 128;

        public List<int> Convert(string path)
        {
            var result = new List<int>();

            var image = new Bitmap(path);
            var resizeImage = new Bitmap(image, new Size(50, 50));
            Height = resizeImage.Height;
            Width = resizeImage.Width;

            for (int y = 0; y < resizeImage.Height; y++)
            {
                for (int x = 0; x < resizeImage.Width; x++)
                {
                    var pixel = resizeImage.GetPixel(x, y);
                    var value = Brightness(pixel);
                    result.Add(value);
                }
            }

            return result;
        }

        private int Brightness(Color pixel)
        {
            var result = 0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B;
            return result < Boundary ? 0 : 1;
        }

        public void Save(string path, List<int> pixels)
        {
            var image = new Bitmap(Width, Height);
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var color = pixels[x + y * Width] == 1 ? Color.White : Color.Black;
                    image.SetPixel(x, y, color);
                }
            }

            image.Save(path);
        }
    }
}
