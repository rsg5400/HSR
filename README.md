
## MPNet-base-2
![Imgur](https://imgur.com/p9gjdwp.png)

**optimum-cli export onnx --model sentence-transformers/all-mpnet-base-v2 mpnet-base-2**



## All-mini-l6-v2
![Imgur](https://imgur.com/hernQQX.png)

**optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 all-mini-l6-v2**

### Package to add

- Microsoft.ML
- Microsoft.ML.OnnxRuntime

### TODO
1. Add support for batches
2. Look more into what you can do with different model attributes
3. Add position encoding
4. Preprocessing data
5. Retrain model + tokenizer?