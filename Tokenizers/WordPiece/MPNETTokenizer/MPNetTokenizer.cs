
using NLP.Models;

namespace NLP.Tokenizers.WordPiece.MPNETTokenizers
{
    public class MPNetTokenizer : MPNETUncasedTokenizer
    {
        public MPNetTokenizer(IOnnxModel model) : base(model)
        {
        }
    }
}