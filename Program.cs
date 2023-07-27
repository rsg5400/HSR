using System;
using System.Text.Json;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;
using NLP.Helpers;
using NLP.Tokenizers;
using NLP.Models.ModelInputs;
using NLP.Models.ModelConfiguration;
using NLP.Pipelines;

namespace NLP{
    public class Program{
        public static void Main(){
            //creates BERTConfiguration
            var AllMini_l6_v2 = new BertModelConfiguration(modelPath : "/home/sam/models/model.onnx", vocabularyFile : "/home/sam/SentSim/vocab.txt");
            //Creates ONNX RUNTIME session(using the same session multiple times is much more efficient)
            AllMini_l6_v2.CreateSession();
            //Passes BERTCONFIG to build input and predict
            var model = new BERTModel(AllMini_l6_v2);
            var output = model.Predict("This movie is bad");
            Console.WriteLine(output[0]);
            var mpNet2 = new MPNETModelConfig(modelPath : "/home/sam/mpnet-base-2/model.onnx", vocabularyFile : "/home/sam/mpnet-base-2/vocab.txt");
            mpNet2.CreateSession();

            var model1 = new MPNETModel(mpNet2);

            var out1 = model1.Predict("This is great");
            Console.WriteLine(out1[0]);
            //returns a float[] of normalized sentence embeddings
  
            //from here you can call cosine similarity if you have two or more embeddings
          
        }
    }
}