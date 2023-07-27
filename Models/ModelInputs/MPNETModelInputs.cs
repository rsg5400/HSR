using Microsoft.ML.Data;

namespace NLP.Models.ModelInputs{
    public class MPNETModelInputs{

        [ColumnName("input_ids")]
        public long[] InputIds { get; set; }


        [ColumnName("attention_mask")]
        public long[] AttentionMask { get; set; }


    }
}