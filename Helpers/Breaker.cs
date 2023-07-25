using MathNet.Numerics.LinearAlgebra;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SentSim.Helpers{
    public class Breaker{
        public static List<float[]> SplitArray(float[] inputArray, int chunkSize)
        {
            List<float[]> chunks = new List<float[]>();

            for (int i = 0; i < inputArray.Length; i += chunkSize)
            {
                int chunkLength = Math.Min(chunkSize, inputArray.Length - i);
                float[] chunk = new float[chunkLength];
                Array.Copy(inputArray, i, chunk, 0, chunkLength);
                chunks.Add(chunk);
            }


            return chunks;
    }
    }
}