using System.Net.NetworkInformation;
using System.Reflection.PortableExecutable;
using System.Runtime.Intrinsics.Arm;

namespace tester{
    public static class MLMath{
        public static float CosineSimilarity (float[] vectorA, float[] vectorB){
            /*
                cos(theta) = (a dot b)/(mag(a)*mag(b))
            */
            float dotProduct = Dp(vectorA, vectorB);
            float magA = (float)Math.Sqrt(Dp(vectorA, vectorA));
            float magB = (float)Math.Sqrt(Dp(vectorB, vectorB));

            return dotProduct/(magA*magB);


            
        }

        public static float Dp(float[] vectorA, float[] vectorB){
            return vectorA.Zip(vectorB, (a, b) => a * b)
                        .Sum();
        }
    }
}