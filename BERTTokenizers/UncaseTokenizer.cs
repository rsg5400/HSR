using SentSim.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;


namespace SentSim.BERTTokenizers
{
    public abstract class UncasedTokenizer : TokenizerBase
    {
        protected UncasedTokenizer(string vocabularyFilePath) : base(vocabularyFilePath)
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