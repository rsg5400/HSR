using Microsoft.ML.OnnxRuntime;

namespace NLP.Models{
    public interface IOnnxModel{

        public int MaxSequenceLength { get; set; }
        public string VocabularyFile { get; set; }
        public string ModelPath { get; set; }
        public string ModelName {get; set; }

        public Dictionary<string, string> Tokens {get; set;}

        public InferenceSession Session {get; set;}


        
    }
}
    

