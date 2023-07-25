using SentSim.Extensions;
using SentSim.Helpers;
using SentSim.MachineLearning;
using SentSim.MachineLearning.MLModel;
using SentSim.BERTTokenizers;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection.Metadata.Ecma335;
using System.Runtime.InteropServices;

namespace SentSim
{
    public class SimModel
    {
        private List<string> _vocabulary;

        private readonly Tokenizer _tokenizer;
        private Predictor _predictor;

        public SimModel(string vocabularyFilePath, string bertModelPath)
        {
            _vocabulary = FileReader.ReadFile(vocabularyFilePath);
            _tokenizer = new Tokenizer(vocabularyFilePath);

            var trainer = new Trainer();
            var trainedModel = trainer.BuildandTrain(bertModelPath, false);
            _predictor = new Predictor(trainedModel);
        }

        public float[] Predict(InferenceSession session, params string[] sentence)
        {


            
            var input = BuildInput(sentence);

            var attentionMask = input.ToArray()[1].AsTensor<long>();
            
            var output = session.Run(input);
            var other = output.First().AsEnumerable<float>().ToArray();
            var outputTensor = output.First().AsTensor<float>();
            ReadOnlySpan<int> dim = output.First().AsTensor<float>().Dimensions;
            
                         
            var final = SentenceEmbeddingGenerator.MeanPoolingWithAttention(outputTensor, attentionMask);

           
            var final1 = SentenceEmbeddingGenerator.Normalize(final, 1);
            return final1;

        }
      
        //This function builds the inputs that OnnxRuntime takes in
        private List<Microsoft.ML.OnnxRuntime.NamedOnnxValue> BuildInput(params string[] sentence)
        {
            //creates tokens from sentence input
            var tokens = _tokenizer.Tokenize(sentence);

            /* ex: "This movie is bad"
                tokens = [("[CLS]",101,0),("this",2023,0), etc]
            */

            var encoded = _tokenizer.Encode(tokens, sentence);
            /* ex:   [CLS] this movie is bad [SEP]
            encoded =  [(101,0,1), (2023,0,1), (3185,0,1), (2003,0,1), (2919,0,1), (102,0,1)]
            */

           //creates model input from encoded
            var modelInput = new ModelInput()
                {
                    InputIds = encoded.Select(t => t.InputIds).ToArray(),
                    AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                    TokenTypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
                };
            //converts inputs into tensors using Tensors class in SentSim.Helpers
            var input_ids = Tensors.ConvertToTensor(modelInput.InputIds, modelInput.InputIds.Length);
            /* input_ids = {{101,2023,3185,2003,2919,102}} for "this movie is bad*/
            var attention_mask = Tensors.ConvertToTensor(modelInput.AttentionMask, modelInput.InputIds.Length);
            var token_type_ids = Tensors.ConvertToTensor(modelInput.TokenTypeIds, modelInput.InputIds.Length);



            /* converts tensors into NamedOnnxValues which adds metadeta like Tensor name 
                and is what OnnxRuntime takes in as input*/
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", input_ids), 
                                         NamedOnnxValue.CreateFromTensor("attention_mask", attention_mask), 
                                         NamedOnnxValue.CreateFromTensor("token_type_ids", token_type_ids) };

            return input;
        }

    

             

    }
}