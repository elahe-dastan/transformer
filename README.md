# Transformers
Nowadays, transformers are the hittest topic that everyone is whispering about. Any new models and approaches either use 
transformers themselves or use the idea in them, so I felt my github sea is empty without this fish :D. I'm going to 
simply go through [Transformer model for language understanding](https://www.tensorflow.org/text/tutorials/transformer) 
tutorial.

# Project Goal
We want to train a transformer model to translate a Portuguese to English dataset. We try to code mostly from scratch 
but can be minimized taking advantage of built-in APIs like tf.keras.layers.MultiHeadAttention.

# Advantages vs Disadvantages
It uses stacks of self-attention layers instead of RNNs or CNNs.

### Upsides
1. It makes no assumption about temporal/spatial relationships across the data.
2. Parallelizable (in contrast to RNN)
3. It can learn long-range dependencies.

### Downsides
1. For a time-series, the output for a time-step is calculated from the entire history instead of only the inputs and 
current hidden-state. This may be less efficient.
2. If the input does have a temporal/spatial relationship, like text, some positional encoding must be added or the 
model will effectively see a bag of words.

# Dataset
The dataset is Portuguese-English translation dataset which contains approximately 50000 training examples, 1100 
validation examples, and 2000 test examples.

# Text tokenization & detokenization
The text needs to be converted to some numeric representation first. Typically, you convert the text to sequences of 
token IDs, which are used as indices into an embedding. One popular implementation builds subword tokenizers 
(text.BertTokenizer) optimized for this dataset and exports them in a saved_model.