using SentSim.Extensions;
using SentSim.Helpers;
using SentSim.MachineLearning;
using SentSim.MachineLearning.ModelInputs;
using SentSim.BERTTokenizers;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection.Metadata.Ecma335;
using System.Runtime.InteropServices;
using SentSim.ModelTypes;

namespace SentSim
{
    public class SimModel
    {
        private readonly BertModelConfiguration _bertModelConfiguration;
        private readonly Tokenizer _tokenizer;
     
        public SimModel(BertModelConfiguration bertModelConfiguration)
        {
            _bertModelConfiguration = bertModelConfiguration;
            //_bertModelConfiguration.CreateSession();
            _tokenizer = new Tokenizer(_bertModelConfiguration.VocabularyFile);
        }

        /*This funciton takes in a string(which can be an array of string as well) and returns
          a normalized sentence embedding*/
        public float[] Predict(params string[] sentence)
        {
            //builds input for model 
            var input = BuildInput(sentence);
            /*Runs the model and returns a List<IDisposableNamedOnnxValues 
              and it gets converted to a tensor*/
            var modelOutputTensor = _bertModelConfiguration.Run(input).First().AsTensor<float>(); 
            /*we grab the attention mask tensor in order to perform mean pooling*/
            var attentionMaskTensor = input.ToArray()[1].AsTensor<long>();

            var sentenceEmbeddings = SentenceEmbeddingGenerator.MeanPoolingWithAttention(modelOutput : modelOutputTensor, attentionMask : attentionMaskTensor);

           
            var final1 = SentenceEmbeddingGenerator.Normalize(input : sentenceEmbeddings, p : 2, dim : 1);
            return final1;

        }
      
        //This function builds the inputs that OnnxRuntime takes in
        private List<Microsoft.ML.OnnxRuntime.NamedOnnxValue> BuildInput(params string[] sentence)
        {
            //creates tokens from sentence input
            var tokens = _tokenizer.Tokenize(texts : sentence);
            /* ex: "This movie is bad"
                tokens = [("[CLS]",101,0),("this",2023,0), etc]
            */

            var encoded = _tokenizer.Encode(tokens : tokens);
            /* ex:   [CLS] this movie is bad [SEP]
            encoded =  [(101,0,1), (2023,0,1), (3185,0,1), (2003,0,1), (2919,0,1), (102,0,1)]
            */

           //creates model input from encoded
            var modelInput = new BERTModelInput()
                {
                    InputIds = encoded.Select(t => t.InputIds).ToArray(),
                    AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                    TokenTypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
                };
            //converts inputs into tensors using Tensors class in SentSim.Helpers
            var input_ids = Tensors.ConvertToTensor(inputArray : modelInput.InputIds, inputDimension : modelInput.InputIds.Length);
            /* input_ids = {{101,2023,3185,2003,2919,102}} for "this movie is bad*/
            var attention_mask = Tensors.ConvertToTensor(inputArray : modelInput.AttentionMask, inputDimension : modelInput.InputIds.Length);
            var token_type_ids = Tensors.ConvertToTensor(inputArray : modelInput.TokenTypeIds, inputDimension : modelInput.InputIds.Length);
            
            /* converts tensors into NamedOnnxValues which adds metadeta like Tensor name 
                and is what OnnxRuntime takes in as input*/
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name : "input_ids", value : input_ids), 
                                         NamedOnnxValue.CreateFromTensor(name : "attention_mask", value : attention_mask), 
                                         NamedOnnxValue.CreateFromTensor(name : "token_type_ids", value : token_type_ids) };

            return input;
        }
    }
}