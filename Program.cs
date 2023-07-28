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
            // var AllMini_l6_v2 = new BertModelConfiguration(modelPath : "/home/sam/models/model.onnx", vocabularyFile : "/home/sam/SentSim/vocab.txt");
            // //Creates ONNX RUNTIME session(using the same session multiple times is much more efficient)
            // AllMini_l6_v2.CreateSession();
            // //Passes BERTCONFIG to build input and predict
            // var model1 = new BERTModel(AllMini_l6_v2);
            // var output = model.Predict("This movie is bad");
            // Console.WriteLine(output[0]);
             var mpNet2 = new MPNETModelConfig(modelPath : "/home/sam/mpnet-base-2/model.onnx", vocabularyFile : "/home/sam/mpnet-base-2/vocab.txt");
             mpNet2.CreateSession();

             var model1 = new MPNETModel(mpNet2);

             var sentences = "This movie is bad.";

             var out1 = model1.Predict(sentences);
             Console.WriteLine(out1[0]);
//             List<(string, float[])> list = new List<(string, float[])>();

//             string sentence1 = "The movie oppenheimer was much better than Barbie";
//             string sentence2 = "The Dell XPS 13 is the greatest laptop of all time";
//             string sentence3 = "Ohio state is a horrible school";

//             var out1 = model1.Predict("This is the best thing");

//             list.Add((sentence1, model1.Predict(sentence1)));
//             list.Add((sentence2, model1.Predict(sentence2)));
//             list.Add((sentence3, model1.Predict(sentence3)));
//  // Create a new instance of Stopwatch
//             Stopwatch stopwatch = new Stopwatch();

//         // Start the stopwatch
//             stopwatch.Start();

//             var useEmbed = model1.Predict("There are a lot of good machines for computing especially that DELl one");



//             string final = EmbeddingsCalc.FindClosest(useEmbed, list);
//             stopwatch.Stop();

//             Console.WriteLine(stopwatch.Elapsed.Milliseconds);
            // Console.WriteLine(out1[0]);

            //returns a float[] of normalized sentence embeddings

            //from here you can call cosine similarity if you have two or more embeddings

        }
    }
}