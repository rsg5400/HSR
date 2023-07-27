using Microsoft.ML.OnnxRuntime.Tensors;

namespace NLP.Helpers{
    public class SentenceEmbeddingGenerator
    {
        /*Creates sentence embeddings from word embeddings and attention mask*/
        public static float[] MeanPoolingWithAttention(Tensor<float> modelOutput, Tensor<long> attentionMask)
        {
            //turns attention mask into float array
            long[] attentionMaskArr = attentionMask.ToArray();
            /*word embeddings is an array of tokens*embeddingsLen*/
            float[] wordEmbeddings = modelOutput.ToArray();
            ReadOnlySpan<int> dimesions = modelOutput.Dimensions;
            //number of tokens/words
            int numTokens = dimesions[1];
            //size of each word embedding
            int embeddingsLen = dimesions[2];
            float[] sentenceEmbedding = new float[embeddingsLen];


            for (int i = 0; i < numTokens; i++)
            {
                //grabs attention mask for that token
                double weight = attentionMaskArr[i];
                for (int j = 0; j < embeddingsLen; j++)
                {
                    /*Adds equaivalent value for each token into a sentence embedding. Since wordEmbeddings is 
                      a one dimesional array that has multiple different words you can think of this like being
                      wordEmbeddings[i][j] if it had been a two dimesional array*/
                    sentenceEmbedding[j] += (float)(wordEmbeddings[(embeddingsLen*i)+j] * weight);
                }
            }


            long totalAttention = attentionMask.Sum();
            for (int i = 0; i < embeddingsLen; i++)
            {
                //average each value for sentence embeddings
                sentenceEmbedding[i] /= totalAttention;
            }
            
            return sentenceEmbedding;
        }

        public static float[] Normalize(float[] input, float p = 2, int dim = 1, double eps = 1e-9)
        {
            /*  
            Lp = (v1^p + v2^p + v3^p)**1/p  
            */

            // Calculate Lp norm along the specified dimension
            double sumSquares = 0.0;
            for (int i = 0; i < input.Length; i++)
            {
                sumSquares += Math.Pow(input[i],p);
            }

            double sqrtSumSquares = Math.Pow(sumSquares, 1/p);

            // Avoid division by zero
            
            double norm = Math.Max(sqrtSumSquares, eps);

            // Normalize the input array
            float[] normalizedArray = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                normalizedArray[i] = (float)(input[i] / norm);
            }

            return normalizedArray;
    }
}

    }
