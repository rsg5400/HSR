using Microsoft.ML.Data;

namespace NLP.Models.ModelInputs{
    public class BERTModelInput{

        [ColumnName("input_ids")]
        public long[] InputIds { get; set; }


        [ColumnName("attention_mask")]
        public long[] AttentionMask { get; set; }


        [ColumnName("token_type_ids")]
        public long[] TokenTypeIds { get; set; }
    }
}