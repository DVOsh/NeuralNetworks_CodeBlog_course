using NeuralNetwork_CodeBlog_course;

namespace NeuralNetwork_Tests;

[TestClass]
public class PictureConverterTests
{
    [TestMethod]
    public void ConvertTest()
    {
        var converter = new PictureConverter();

        var inputs = converter.Convert(@"D:\Coding\CSharp\Education\CodeBlog\NeuralNetwork_CodeBlog_course\NeuralNetwork_Tests\Images\Parasitized.png");
        converter.Save("d:\\image.png", inputs);

        //var input1 = converter.Convert(@"D:\Coding\CSharp\Education\CodeBlog\Datasets\cell_images\Parasitized\C33P1thinF_IMG_20150619_114756a_cell_179.png");
        //var input2 = converter.Convert(@"D:\Coding\CSharp\Education\CodeBlog\Datasets\cell_images\Parasitized\C33P1thinF_IMG_20150619_114756a_cell_180.png");

        //Console.WriteLine(input1.Count);
        //Console.WriteLine(input2.Count);
    }
}
