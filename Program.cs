using System;
using System.Text.Json;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;
using SentSim.Helpers;
using SentSim.BERTTokenizers;
using SentSim.MachineLearning.ModelInputs;
using tester;
using SentSim.ModelTypes;

namespace SentSim{
    public class Program{
        public static void Main(){
            //creates BERTConfiguration
            var AllMini_l6_v2 = new BertModelConfiguration(modelPath : "/home/sam/models/model.onnx", vocabularyFile : "/home/sam/SentSim/vocab.txt");
            //Creates ONNX RUNTIME session(using the same session multiple times is much more efficient)
            AllMini_l6_v2.CreateSession();
            //Passes BERTCONFIG to build input and predict
            var model = new SimModel(AllMini_l6_v2);

            //returns a float[] of normalized sentence embeddings
            var output = model.Predict("This movie is bad");
            
            //from here you can call cosine similarity if you have two or more embeddings
          
        }
    }
}