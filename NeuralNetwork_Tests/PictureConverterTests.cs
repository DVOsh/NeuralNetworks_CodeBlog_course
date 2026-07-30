using NeuralNetwork_CodeBlog_course;

namespace NeuralNetwork_Tests;

[TestClass]
public class PictureConverterTests
{
    [TestMethod]
    public void ConvertTest()
    {
        var converter = new PictureConverter();
        var inputs = converter.Convert(@"D:\Coding\CSharp\Education\CodeBlog\NeuralNetwork_CodeBlog_course\NeuralNetwork_Tests\Images\Uninfected.png");
        converter.Save("d:\\image.png", inputs);
    }
}
