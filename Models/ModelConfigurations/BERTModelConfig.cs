using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualBasic;
using NLP.Models;

namespace NLP.Models.ModelConfiguration{
    public class BertModelConfiguration : IOnnxModel
    {
        public int MaxSequenceLength { get; set; }
        public string VocabularyFile { get; set; }
        public string ModelPath { get; set; }
        public string ModelName {get; set; }

        public Dictionary<string, string> Tokens {get; set;}


     
        public InferenceSession Session {get; set;}

        public BertModelConfiguration(string modelPath, string vocabularyFile,int maxSequenceLength = 256, string modelName = "BERT Model"){
            this.MaxSequenceLength = maxSequenceLength;
            this.ModelPath = modelPath;
            this.VocabularyFile = vocabularyFile;
            this.ModelName = modelName;

            Tokens = new Dictionary<string, string>{
                {"Unknown", "[UNK]"},
                {"Separation", "[SEP]"}, 
                {"Padding", ""}, 
                {"Classification", "[CLS]"}
            };


        }

        public void CreateSession(){
            this.Session = new InferenceSession(this.ModelPath);
        } 

        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(List<NamedOnnxValue> input){
            return this.Session.Run(input);
        }
        
    }
}