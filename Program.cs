using System;
using System.Text.Json;
using Microsoft.ML.Data;
using SentSim.MachineLearning.MLModel;

namespace SentSim{
    public class Program{
        public static void Main(string[] args){
            var model = new SimModel("/home/sam/SentSim/vocab.txt",
                                "/home/sam/models/model.onnx");

            ModelOutput output = model.Predict("This movie was good.");

            ModelOutput output1 = model.Predict("This movie was great.");
             
            for(int i = 0; i < 1000; i++){
                model.Predict("This movie was great.");
            }

            
            var random = output.LastHiddenState.GetValues();
            var random1 = output1.LastHiddenState.GetValues();

            

        }
    }
}