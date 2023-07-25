using Microsoft.ML.OnnxRuntime.Tensors;

namespace SentSim.Helpers{
    public class SentenceEmbeddingGenerator
    {
        // Function to perform mean pooling on word embeddings with attention mask
        public static float[] MeanPoolingWithAttention(Tensor<float> modelOutput, Tensor<long> attentionMask)
        {
            long[] attentionMaskArr = attentionMask.ToArray();
            float[] wordEmbeddings = modelOutput.ToArray();
            ReadOnlySpan<int> dimesions = modelOutput.Dimensions;
            int tokens = dimesions[1];
            int embeddingsLen = dimesions[2];
            float[] sentenceEmbedding = new float[embeddingsLen];


            for (int i = 0; i < tokens; i++)
            {
                double weight = attentionMaskArr[i];
                for (int j = 0; j < embeddingsLen; j++)
                {
                    sentenceEmbedding[j] += (float)(wordEmbeddings[(embeddingsLen*i)+j] * weight);
                }
            }


            long totalAttention = attentionMask.Sum();
            for (int i = 0; i < embeddingsLen; i++)
            {
                sentenceEmbedding[i] /= totalAttention;
            }
            
            return sentenceEmbedding;
        }

        public static float[] Normalize(float[] input, int dimension = 0)
        {
            if (input == null || input.Length == 0)
            {
                throw new ArgumentException("Input array cannot be null or empty.");
            }

            int rows, cols;
            if (dimension == 0)
            {
                rows = input.Length;
                cols = 1;
            }
            else
            {
                rows = 1;
                cols = input.Length;
            }

            // Calculate L2 norm along the specified dimension
            double sumSquares = 0.0;
            for (int i = 0; i < input.Length; i++)
            {
                sumSquares += input[i] * input[i];
            }

            double sqrtSumSquares = Math.Sqrt(sumSquares);

            // Avoid division by zero
            double epsilon = 1e-9;
            double norm = Math.Max(sqrtSumSquares, epsilon);

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
