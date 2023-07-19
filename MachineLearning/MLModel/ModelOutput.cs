using System.Numerics;
using Microsoft.ML.Data;

namespace SentSim.MachineLearning.MLModel{
    public class ModelOutput{
        [ColumnName("last_hidden_state")]
        public VBuffer<float> LastHiddenState { get; set; }
    }
}