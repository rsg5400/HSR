using System;
using System.Text.Json;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using SentSim.Helpers;
using SentSim.BERTTokenizers;
using SentSim.MachineLearning.MLModel;

namespace SentSim{
    public class Program{
        public static void Main(string[] args){
            var model = new SimModel("/home/sam/SentSim/vocab.txt",
                                "/home/sam/models/model.onnx");

            var session = new InferenceSession("/home/sam/models/model.onnx");
            
            
          
            var output = model.Predict("This movie was good.", session);

            var outputArray = output?.ToArray();
            
            //float[] embeddings = outputArray[0].AsEnumerable<float>().ToArray();


            //var inference = output?.ToList().First().AsDictionary<string, float>(); 

            //Console.WriteLine(Tensors.ConvertToTensor(output[0])[0]);
            
        }
    }
}