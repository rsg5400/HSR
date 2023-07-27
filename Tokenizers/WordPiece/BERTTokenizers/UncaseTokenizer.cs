using NLP.Extensions;
using NLP.Models;
using System;
using System.Collections.Generic;
using System.Linq;


namespace NLP.Tokenizers.WordPiece.BERTTokenizers
{
    public abstract class UncasedTokenizer : BERTTokenizerBase
    {
        protected UncasedTokenizer(IOnnxModel model) : base(model)
        {
        }

        protected override IEnumerable<string> TokenizeSentence(string text)
        {
            return text.Split(new string[] { " ", "   ", "\r\n" }, StringSplitOptions.None)
                .SelectMany(o => o.SplitAndKeep(".,;:\\/?!#$%()=+-*\"'–_`<>&^@{}[]|~'".ToArray()))
                .Select(o => o.ToLower());
            
        }
    }
}