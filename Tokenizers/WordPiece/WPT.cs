using System.Text.RegularExpressions;
using NLP.Helpers;
using NLP.Models;

namespace NLP.Tokenizers.WordPiece{
    public abstract class WordPieceTokenizer{
        
        protected readonly List<string> _vocabulary;
        protected readonly Dictionary<string, int> _vocabularyDict;
        protected readonly IOnnxModel _model;
      
        public WordPieceTokenizer(IOnnxModel model)
        {
            _model=model;
            _vocabulary = FileReader.ReadFile(_model.VocabularyFile);

            _vocabularyDict = new Dictionary<string, int>();
            for (int i = 0; i < _vocabulary.Count; i++)
                _vocabularyDict[_vocabulary[i]] = i;
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



        protected IEnumerable<long> SegmentIndex(List<(string token, int index)> tokens)
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

                if (token == _model.Tokens["Separation"])
                {
                    segmentIndex++;
                }
            }

            return segmentIndexes;
        }

        protected IEnumerable<(string Token, int VocabularyIndex)> TokenizeSubwords(string word)
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
                    tokens.Add((_model.Tokens["Unknown"], _vocabularyDict[_model.Tokens["Unknown"]]));

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
                tokens.Add((_model.Tokens["Unknown"], _vocabularyDict[_model.Tokens["Unknown"]]));
            }

            return tokens;
        }

        protected abstract IEnumerable<string> TokenizeSentence(string text);
    }
}
