using NeuralNetwork_CodeBlog_course;

namespace MedicalSystem
{
    public partial class MainForm : Form
    {
        public MainForm()
        {
            InitializeComponent();
        }

        private void aboutToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var aboutForm = new AboutForm();
            aboutForm.ShowDialog();
        }

        private void imageToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                var pictureConverter = new PictureConverter();
                var inputs = pictureConverter.Convert(openFileDialog.FileName);
                var result = Program.Controller.ImageNetwork.FeedForward();
            }
        }

        private void enterToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var enterDataForm = new EnterData();
            var result = enterDataForm.ShowForm();
        }
    }
}
