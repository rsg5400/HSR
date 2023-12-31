using Microsoft.ML.OnnxRuntime.Tensors;

namespace NLP.Helpers{
    public static class Tensors{

    
        public static Tensor<long> ConvertToTensor(long[] inputArray, int inputDimension, int batchNum = 1)
        {
            /* creates Rank 2 Tensor with length equaling the amount of tokens.
                Dimesion in equavalent to batch size which with one sentence is one.
            */

            Tensor<long> input = new DenseTensor<long>(new[] { batchNum, inputDimension });

            // Loop through the inputArray (InputIds, AttentionMask and TypeIds)
            for (var i = 0; i < inputArray.Length; i++)
            {
                /* adding input value to Tensor. input[batchNum,tokenIndex] = input[tokenIndex] */
                input[0,i] = inputArray[i];
            }
           // Console.WriteLine(input.Length);
            return input;
        }
    }
}