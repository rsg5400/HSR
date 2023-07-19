using SentSim.Extensions;
using SentSim.Helpers;
using SentSim.MachineLearning;
using SentSim.MachineLearning.MLModel;
using SentSim.BERTTokenizers;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

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

        public ModelOutput Predict(string sentence)
        {
            
            var tokens = _tokenizer.Tokenize(sentence);
            
            var input = BuildInput(tokens); 
            
            var predictions = _predictor.Predict(input);


            return predictions;

            //var contextStart = tokens.FindIndex(o => o.Token == Token.Separation);

            // predictions hold the embedings and now I need to compute cosine similarity

           // var predictedTokens = input.InputIds
           //     .Skip(startIndex)
           //     .Take(endIndex + 1 - startIndex)
          //      .Select(o => _vocabulary[(int)o])
          //      .ToList();

            //var connectedTokens = _tokenizer.Untokenize(predictedTokens);

           // return (connectedTokens, probability);
        }
       // public float CosineSimilarity(double[] VectorA, double[] VectorB){
        //    IEnumerable<float> dotProduct = CosineSim.ComputeDotProduct(VectorA, VectorB);

       // }

        private ModelInput BuildInput(List<(string Token, int Index, long SegmentIndex)> tokens)
        {
            var padding = Enumerable.Repeat(0L, 256 - tokens.Count).ToList();

            var tokenIndexes = tokens.Select(token => (long)token.Index).Concat(padding).ToArray();
            var segmentIndexes = tokens.Select(token => token.SegmentIndex).Concat(padding).ToArray();
            var inputMask = tokens.Select(o => 1L).Concat(padding).ToArray();

            return new ModelInput()
            {
                InputIds = tokenIndexes,
                TokenTypeIds = segmentIndexes,
                AttentionMask = inputMask,
            };
        }

        
    }
}