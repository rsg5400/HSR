using Microsoft.ML;
using SentSim.MachineLearning.MLModel;

namespace SentSim.MachineLearning{
    public class Predictor{
        private MLContext _mLContext;
        private PredictionEngine<ModelInput, ModelOutput> _predictionEngine;

        public Predictor(ITransformer trainedModel){
            _mLContext = new MLContext();
            _predictionEngine = _mLContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
        }

        public ModelOutput Predict(ModelInput encodedInput)
        {
            return _predictionEngine.Predict(encodedInput);
        }
    }
}