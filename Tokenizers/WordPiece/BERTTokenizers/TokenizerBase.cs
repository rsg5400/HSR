using NLP.Helpers;
using NLP.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace NLP.Tokenizers.WordPiece.BERTTokenizers
{
    public abstract class BERTTokenizerBase : WordPieceTokenizer
    {
        
   
        public BERTTokenizerBase(IOnnxModel model) : base(model)
        {
   
        }

        
        public List<(long InputIds, long TokenTypeIds, long AttentionMask)> Encode(List<(string Token, int VocabularyIndex, long SegmentIndex)> tokens)
        {
            /* if you have multiple batches of different lengths you need to set sequence legnth
            to either the length of largest tokenizer sequence or to some max sequene constant like
            256*/

            int sequenceLength = tokens.Count;

            var padding = Enumerable.Repeat(0L, sequenceLength - tokens.Count).ToList();
            /*concats padding onto input_ids input(When embedding one sentence no padding it added)
                       ex:   [CLS] this movie is bad [SEP]
              tokenIndexes =  101  2023 3185  2003 2919 102
              */
            var tokenIndexes = tokens.Select(token => (long)token.VocabularyIndex).Concat(padding).ToArray();
            /*concats padding onto token_type_ids input(When embedding one sentence no padding it added)
                       ex:   [CLS] this movie is bad [SEP]
              segmentIndexes =  0    0   0    0   0    0           
            */
        
            var segmentIndexes = tokens.Select(token => token.SegmentIndex).Concat(padding).ToArray();
            /* this represents the attention_mask. Attention mask has a value of 1 for every token that is not a padding
               so if you are encoding only one sentence the attention_mask with be an array filled with 1s
                       ex:   [CLS] this movie is bad [SEP]
              tokenIndexes =  1      1   1     1  1    1               
               */
            var inputMask = tokens.Select(o => 1L).Concat(padding).ToArray();

            var output = tokenIndexes.Zip(segmentIndexes, Tuple.Create)
                .Zip(inputMask, (t, z) => Tuple.Create(t.Item1, t.Item2, z));
            /* ex:   [CLS] this movie is bad [SEP]
            return:  [(101,0,1), (2023,0,1), (3185,0,1), (2003,0,1), (2919,0,1), (102,0,1)]
            */
            return output.Select(x => (InputIds: x.Item1, TokenTypeIds: x.Item2, AttentionMask:x.Item3)).ToList();
        }
    

        public List<(string Token, int VocabularyIndex, long SegmentIndex)> Tokenize(params string[] texts)
        {
            // This puts a [CLS] at the beggining of every batch
            IEnumerable<string> tokens = new string[] { _model.Tokens["Classification"] };
            
            //loops through texts(in HSR there is only one value in array)
            foreach (var text in texts)
            {
                //Tokenizer each text in texts(could be more than one sentence as well)
                tokens = tokens.Concat(TokenizeSentence(text));
             /* Puts a [SEP] marker at the end of each sequence*/
                tokens = tokens.Concat(new string[] { _model.Tokens["Separation"] });
            }
            // takes each individual token and returns and IEnumerable(token(actualy word), vocab index(tokens corresponding row number in vocabulary file))
            var tokenAndIndex = tokens
                .SelectMany(TokenizeSubwords)
                .ToList();
            /* ex: [CLS] this movie is bad [SEP]

                tokenAndIndex = [("[CLS]",101), ("this",2023), etc]
            */

            var segmentIndexes = SegmentIndex(tokenAndIndex);
            
            return tokenAndIndex.Zip(segmentIndexes, (tokenindex, segmentindex)
                                => (tokenindex.Token, tokenindex.VocabularyIndex, segmentindex)).ToList();
        }


        protected override abstract IEnumerable<string> TokenizeSentence(string text);
    }
}
