using Microsoft.ML;
using SentSim.MachineLearning.MLModel;

namespace SentSim.MachineLearning{
    public class Trainer{
        private readonly MLContext _mlContext;

        public Trainer(){
            _mlContext = new MLContext(11);
        }

        public ITransformer BuildandTrain(string modelPath, bool useGpu){
            var pipeline = _mlContext.Transforms
                            .ApplyOnnxModel(modelFile: modelPath, 
                                           shapeDictionary: new Dictionary<string, int[]>
                                            {
                                                { "input_ids", new [] { 1, 256 } },
                                                { "attention_mask", new [] { 1, 256 } },
                                              	{ "token_type_ids", new [] { 1, 256 } },
                                                { "last_hidden_state", new [] { 1, 256, 384 } },
                                            },
                                            outputColumnNames: new[] { "last_hidden_state" }, 
                                            inputColumnNames: new[] {"input_ids",
                                                                "attention_mask",
                                                                "token_type_ids"}, 
                                            gpuDeviceId: useGpu ? 0 : (int?)null,
                                            fallbackToCpu: true);

            return pipeline.Fit(_mlContext.Data.LoadFromEnumerable(new List<ModelInput>()));
        }

    }
}