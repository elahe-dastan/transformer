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

# Positional encoding

Attention layers see their input as a set of vectors, with no sequential order. This model also doesn't contain any 
recurrent or convolutional layers. Because of this a "positional encoding" is added to give the model some information 
about the relative position of tokens in the sentence.

The positional encoding vector is added to the embedding vector. Embeddings represent a token in a d-dimensional space
where tokens with similar meaning will be closer to each other. But the embeddings do not encode the relative position 
of tokens in a sentence. So after adding the positional encoding, tokens will be closer to each other based on the 
**similarity of their meaning and their position in the sentence**, in the d-dimensional space.

The formula for calculating the positional encoding is as follows:

![positional encoding](images/positional_encoding.png)

## Intuition on the formula
We want to bring word order information to transformers. How about we introduce a set of vectors containing the position
information called position embeddings and add them to the previous embedding but what values should our position 
embeddings contain :thinking:

1. Why not simply consider the word position number

![word position](images/word_position.png)

adding the position information like this may significantly distort the embedding information, for example if the text 
has 30 words, the last embedding will be added to the huge number of 30.

2. What if instead we added fractions

![word position fraction](images/word_position_fraction.png)

This way the maximum embedding value will not surpass 1. It doesn't work either because making the position embeddings a 
function of the total text length would mean **if the sentences differ in length they would possess different position 
embeddings for the same position this may in turn confuse the model**

![different length sentences](images/different_length_sentences.png)

Ideally, the position embedding values at a given position should remain the same irrespective of the text or any other 
factor.

3. The authors used wave frequencies to capture position information. Let's take the first position embedding as an 
example therefore the pos variable in the formula will be 0. Next the size of the position embedding has to be the same 
as the word embedding this is represented by the letter d in the formula. The letter i here represents the indices of 
each of the position embedding dimensions

![word position frequency](images/word_position_frequency.png)

### Why does it work?
Now if we plot a sinusoidal curve by varying the variable indicating board positions on the x-axis we will get a smooth 
looking curve, **since the curve height only varies between a fixed range and is not dependent on the text length**, 
this method can help us overcome the limitation previously discussed.

![sin curve](images/sin_curve.png)

There is a problem though, note the embeddings of position 0 and 6 are exactly the same. This is when the next variable 
in the equation, the i, comes to rescue. If we plot the curve at different values of i's we get a series of curves with 
different frequencies. Now if you read the value of the position embedding for positions zero and six, for i=4 they will 
be exactly the same but for i = 0 for example they are very different.

![sin curves freq](images/sin_curves_freq.png)

This is a positional encoding curve plotted on a full scale, as to get the value of a position embedding at a certain 
dimension, you can simply read off the chart

![positional encoding diagram](images/positional_encoding_diagram.png)

# Look-Ahead Masking
The look-ahead mask is used to mask the future tokens in a sequence. In other words, the mask indicates which entries 
should not be used.

This means that to predict the third token, only the first and second token will be used. Similarly to predict the 
fourth token, only the first, second and the third tokens will be used and so on.

# Scaled Dot-Product Attention

![scaled attention](images/scaled_attention.png)

The attention function used by a transformer takes three inputs: Q(query), K(key), V(value). The equation used to 
calculate the attention weights is:

![Attention formula](images/Attention-formula.png)

The dot-product attention is scaled by a factor of square root of the depth. This is done because for large values of 
depth, the dot product grow large in magnitude pushing the softmax function where it has small gradients resulting in a 
very hard softmax.

For example, consider that "Q" and "K" have a mean of 0 and variance of 1. Their matrix multiplication will have a mean 
of 0 and variance of "dk". So the square root of "dk" is used for scaling, so you get a consistent variance regardless 
of the value of "dk". If the variance is too low the output may be too flat to optimize effectively. If the variance is 
too high the softmax may saturate at initialization making it difficult to learn.

https://www.youtube.com/watch?v=tIvKXrEDMhk
