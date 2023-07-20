using System;
using System.Text.Json;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using SentSim.Helpers;
using SentSim.BERTTokenizers;
using SentSim.MachineLearning.MLModel;
using tester;

namespace SentSim{
    public class Program{
        public static void Main(string[] args){
            var model = new SimModel("/home/sam/SentSim/vocab.txt",
                                "/home/sam/models/model.onnx");

            var session = new InferenceSession("/home/sam/models/model.onnx");
            
            
          
             var output = model.Predict("This Movie was horrible.", session);

             var outputArray = output.ToArray();
            // var output1 = model.Predict("This movie was great.", session);

            // var cs = MLMath.CosineSimilarity(output.ToArray()[0].AsEnumerable<float>().ToArray(), output1.ToArray()[0].AsEnumerable<float>().ToArray());
            // Console.WriteLine(cs.ToString());
            float[] embeddings = outputArray[0].AsEnumerable<float>().ToArray();


            foreach(var item in embeddings){
                Console.WriteLine(item.ToString());
            }
            //var inference = output?.ToList().First().AsDictionary<string, float>(); 

            //Console.WriteLine(Tensors.ConvertToTensor(output[0])[0]);
            
        }
    }
}