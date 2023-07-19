using System.ComponentModel.DataAnnotations;
using System.Numerics;

namespace SentSim.Helpers{
    public static class CosineSim{
        public static IEnumerable<double> ComputeDotProduct(double[] VectorA, double[] VectorB){
            IEnumerable<double> dotproduct = VectorA.Zip(VectorB, (a, b) => a * b);
            return dotproduct;
        }
      
    }
}