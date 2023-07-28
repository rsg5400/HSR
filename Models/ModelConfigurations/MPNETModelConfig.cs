using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualBasic;
using NLP.Models;

namespace NLP.Models.ModelConfiguration{
    public class MPNETModelConfig : IOnnxModel
    {
        public int MaxSequenceLength { get; set; }
        public string VocabularyFile { get; set; }
        public string ModelPath { get; set; }
        public string ModelName {get; set; }

        public Dictionary<string, string> Tokens {get; set;}

        public InferenceSession Session {get; set;}


        /*
                                                            Additional Attributes to Consider
                                                            
        initializer_range (float, optional, defaults to 0.02) – The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

        layer_norm_eps (float, optional, defaults to 1e-12) – The epsilon used by the layer normalization layers.

        relative_attention_num_buckets (int, optional, defaults to 32) – The number of buckets to use for each attention layer.
        
        vocab_size (int, optional, defaults to 30527) – Vocabulary size of the MPNet model. Defines the number of different tokens that can be represented by the inputs_ids passed when calling MPNetModel or TFMPNetModel.

        hidden_size (int, optional, defaults to 768) – Dimensionality of the encoder layers and the pooler layer.

        num_hidden_layers (int, optional, defaults to 12) – Number of hidden layers in the Transformer encoder.

        num_attention_heads (int, optional, defaults to 12) – Number of attention heads for each attention layer in the Transformer encoder.

        intermediate_size (int, optional, defaults to 3072) – Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.

        hidden_act (str or Callable, optional, defaults to "gelu") – The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu", "relu", "silu" and "gelu_new" are supported.

        hidden_dropout_prob (float, optional, defaults to 0.1) – The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

        attention_probs_dropout_prob (float, optional, defaults to 0.1) – The dropout ratio for the attention probabilities.
        */

        public MPNETModelConfig(string modelPath, string vocabularyFile,int maxSequenceLength = 512, string modelName = "BERT Model"){
            this.MaxSequenceLength = maxSequenceLength;
            this.ModelPath = modelPath;
            this.VocabularyFile = vocabularyFile;
            this.ModelName = modelName;
            
            Tokens = new Dictionary<string, string>{
                {"bos_token", "<s>"},
                {"eos_token", "</s>"}, 
                {"Uknown", "[UNK]"},
                {"Separation", "</s>"}, 
                {"Padding", "<pad>"},
                {"Classification", "<s>"}
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