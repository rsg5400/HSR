using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualBasic;
using SentSim.ONNX;

namespace SentSim.ModelTypes{
    public class BertModelConfiguration : IOnnxModel
    {
        public int MaxSequenceLength { get; set; }
        public string VocabularyFile { get; set; }
        public string ModelPath { get; set; }
        public string ModelName {get; set; }

        public InferenceSession session;

        public BertModelConfiguration(string modelPath, string vocabularyFile,int maxSequenceLength = 256, string modelName = "BERT Model"){
            this.MaxSequenceLength = maxSequenceLength;
            this.ModelPath = modelPath;
            this.VocabularyFile = vocabularyFile;
            this.ModelName = modelName;
        }

        public void CreateSession(){
            this.session = new InferenceSession(this.ModelPath);
        } 

        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(List<NamedOnnxValue> input){
            return this.session.Run(input);
        }
        
    }
}