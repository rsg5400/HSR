
using NLP.Models;

namespace NLP.Tokenizers.WordPiece.BERTTokenizers
{
    public class Tokenizer : UncasedTokenizer
    {
        public Tokenizer(IOnnxModel model) : base(model)
        {
        }
    }
}