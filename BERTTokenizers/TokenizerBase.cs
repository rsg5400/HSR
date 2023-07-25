using SentSim.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace SentSim.BERTTokenizers
{
    public abstract class TokenizerBase
    {
        protected readonly List<string> _vocabulary;
        protected readonly Dictionary<string, int> _vocabularyDict;

        public TokenizerBase(string vocabularyFilePath)
        {
            _vocabulary = FileReader.ReadFile(vocabularyFilePath);

            _vocabularyDict = new Dictionary<string, int>();
            for (int i = 0; i < _vocabulary.Count; i++)
                _vocabularyDict[_vocabulary[i]] = i;
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

        public string IdToToken(int id)
        {
            return _vocabulary[id];
        }

        public List<string> Untokenize(List<string> tokens)
        {
            var currentToken = string.Empty;
            var untokens = new List<string>();
            tokens.Reverse();

            tokens.ForEach(token =>
            {
                if (token.StartsWith("##"))
                {
                    currentToken = token.Replace("##", "") + currentToken;
                }
                else
                {
                    currentToken = token + currentToken;
                    untokens.Add(currentToken);
                    currentToken = string.Empty;
                }
            });

            untokens.Reverse();

            return untokens;
        }

        public List<(string Token, int VocabularyIndex, long SegmentIndex)> Tokenize(params string[] texts)
        {
            // This puts a [CLS] at the beggining of every batch
            IEnumerable<string> tokens = new string[] { Token.Classification };
            
            //loops through texts(in HSR there is only one value in array)
            foreach (var text in texts)
            {
                //Tokenizer each text in texts(could be more than one sentence as well)
                tokens = tokens.Concat(TokenizeSentence(text));
             /* Puts a [SEP] marker at the end of each sequence*/
                tokens = tokens.Concat(new string[] { Token.Separation });
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

        private static IEnumerable<long> SegmentIndex(List<(string token, int index)> tokens)
        {
            var segmentIndex = 0;
            var segmentIndexes = new List<long>();

            // This will return add a integer corresponding to what sequence it is on
            /*ex 

              token_input_id  [CLS] this movie is bad [SEP]
              segmentIndex       0     0    0    0  0    0

              token_input_id  [CLS] this movie is bad [SEP] i wish i never saw it [SEP]
              segmentIndex       0    0    0    0  0    0   1  1   1   1    1   1   1

            */ 
            foreach (var (token, index) in tokens)
            {
                segmentIndexes.Add(segmentIndex);

                if (token == Token.Separation)
                {
                    segmentIndex++;
                }
            }

            return segmentIndexes;
        }

        private IEnumerable<(string Token, int VocabularyIndex)> TokenizeSubwords(string word)
        {
            // if word in vocab.txt
            if (_vocabularyDict.ContainsKey(word))
            {
                //[word in original sentence, its row in vocab.txt], ex [("this",2023)]
                return new (string, int)[] { (word, _vocabularyDict[word]) };
            }

            var tokens = new List<(string, int)>();
            var remaining = word;

            while (!string.IsNullOrEmpty(remaining) && remaining.Length > 2)
            {
                string prefix = null;
                int subwordLength = remaining.Length;
                while (subwordLength >= 1) // was initially 2, which prevents using "character encoding"
                {
                    //This loops until a subword in found in vocab.txt, if not found then subword will be a substring of length 0 which is null
                    string subword = remaining.Substring(0, subwordLength);
                    if (!_vocabularyDict.ContainsKey(subword))
                    {
                        subwordLength--;
                        continue;
                    }

                    prefix = subword;
                    break;
                }
                // this checks if a subword was found during the previous while loop
                if (prefix == null)
                {
                    // if a subword was not found it adds an unknown token [UNK]
                    tokens.Add((Token.Unknown, _vocabularyDict[Token.Unknown]));

                    return tokens;
                }

                // this runs when subword is found
                // it adds ## to the start of the subword,ex(##bed) and then searches through vocabulary to find it.
                var regex = new Regex(prefix);
                remaining = regex.Replace(remaining, "##", 1);
                //adds token
                tokens.Add((prefix, _vocabularyDict[prefix]));
            }

            if (!string.IsNullOrWhiteSpace(word) && !tokens.Any())
            {
                //returns [("[UNK]",vocabRow)]
                tokens.Add((Token.Unknown, _vocabularyDict[Token.Unknown]));
            }

            return tokens;
        }

        protected abstract IEnumerable<string> TokenizeSentence(string text);
    }
}
