using System.Numerics;
using Microsoft.ML.Data;

namespace SentSim.MachineLearning.MLModel{
    public class ModelOutput{
        [VectorType(1,256,384)]
        [ColumnName("last_hidden_state")]
        public VBuffer<Single> LastHiddenState { get; set; }
    }
}