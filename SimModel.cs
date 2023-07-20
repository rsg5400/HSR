using SentSim.Extensions;
using SentSim.Helpers;
using SentSim.MachineLearning;
using SentSim.MachineLearning.MLModel;
using SentSim.BERTTokenizers;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using System.Reflection.Metadata.Ecma335;

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

        public IEnumerable<DisposableNamedOnnxValue> Predict(string sentence, InferenceSession session)
        {


            
            var input = BuildInput(sentence);

            var output = session.Run(input);


            //var final = ProcessOutput(output1, tokens);

            return output;

        }
       // public float CosineSimilarity(double[] VectorA, double[] VectorB){
        //    IEnumerable<float> dotProduct = CosineSim.ComputeDotProduct(VectorA, VectorB);

       // }

        private List<Microsoft.ML.OnnxRuntime.NamedOnnxValue> BuildInput(string sentence)
        {
            
            var tokens = _tokenizer.Tokenize(sentence);

            var encoded = _tokenizer.Encode(tokens.Count, sentence);

            var modelInput = new ModelInput()
                {
                    InputIds = encoded.Select(t => t.InputIds).ToArray(),
                    AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                    TokenTypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
                };

            var input_ids = Tensors.ConvertToTensor(modelInput.InputIds, modelInput.InputIds.Length);
            var attention_mask = Tensors.ConvertToTensor(modelInput.AttentionMask, modelInput.InputIds.Length);
            var token_type_ids = Tensors.ConvertToTensor(modelInput.TokenTypeIds, modelInput.InputIds.Length);

            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", input_ids), 
                                         NamedOnnxValue.CreateFromTensor("attention_mask", attention_mask), 
                                         NamedOnnxValue.CreateFromTensor("token_type_ids", token_type_ids) };

            return input;
        }

        // public List<string> ProcessOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, List<(string Token, int VocabularyIndex, long SegmentIndex)> tokens){
                


        //     }

             

    }
}