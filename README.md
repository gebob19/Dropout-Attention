# Layer-Specific Dropout with Attention
Brennan Gebotys

# Introduction

This project investigates a new use of self-attention, which to the best of my knowledge is first of its kind. 

## Self-Attention

The Transformer [2] has led to significant advances in Natural Language Processing (NLP) tasks.

Its self-attention involves linear projections of vectors representing a Query, a Key, and a Value. Following the projections is a linear summation weighted by the cosine distance between the projected vectors. The summation is then concatenated with the projected vector and is the projected again from 2*hidden size to hidden size. For self-attention, the Query, Key and Value are all the same vector. 

One significant downfall of the Transformer's attention is interpretability. Given n-words, an L-layer Transformer requires analysis of L * n^2 elements. With our technique, only L * n elements are required to be analyzed.  

## Dropout

Dropout [3] is a technique where units are dropped stochastically with a P(drop) at training time. It is used as a regularization technique for neural networks and is used in most architectures.

A disadvantage of Dropout is that units are dropped entirely at random. When training word-embeddings, Dropout can lead to a significant loss of latent-space information. We show this loss in latent-space information can hinder training larger models significantly. 

Our technique improves model training in less time and we show we are able to train larger models while maintaining validation accuracy without standard Dropout. 

# Algorithm

Assume the input is of a single batch of size (*sequence_length*, *hidden*). Each vector across the sequence length will be refered to as a word-vector.

Each layer in the network will have a corresponding vector (initialized randomly) of size *hidden*. This vector will be called the layer-vector. 

Vector dot-product will be performed with the layer-vector and each word-vector. This operation will result in a vector of size (*sequence_length*, 1). This vector will be called the attention-vector. 

The intuition is that each value in the attention-vector will correspond to how relevant the corresponding word-vector is to the current layer. 

Using the attention-vector we create a probability distribution to stochastically sample the least relevant word-vectors from. 

Let P(i) be the probability of setting all the units at index i of the sequence to zero. 

P(i) = max(attention-vector) - *attention[i]*

To satisfy the probability axioms we apply the softmax function across P(i). We then sample `max(1, sequence_length * dropout_rate)` indices, and set the corresponding word-vector to all zeros.

# Setup 

The dataset used was the Large Movie Review Dataset [1]. The Transformer model [2] was used for tests since it has shown great results in previous Natural Language Processing (NLP) tasks. The Dropout-Attention layers were included after the Multihead-Attention and Feedforward layers. 

# Results

## 6-Layer Transformer (1 Epoch - Full Dataset)
> Test Accuracy <br/>
> Attention Dropout: 0.8494 <br/>
> Dropout: 0.8354 

### Training
![alt text](https://github.com/gebob19/Dropout-Attention/tree/master/images/6layer-train.png) 
### Validation
![alt text](https://github.com/gebob19/Dropout-Attention/tree/master/images/6layer-validation.png)

## 12-Layer Transformer (1 Epoch - Full Dataset)
> Test Accuracy: <br/>
> Attention Dropout:    0.8489 <br/>
> Dropout:              0.4865 

### Training
![alt text](https://github.com/gebob19/Dropout-Attention/tree/master/images/12layer-train.png)

### Validation
![alt text](https://github.com/gebob19/Dropout-Attention/tree/master/images/12layer-validation.png)

## 1-Layer Transformer (1 Epoch - 5K Subset)
> Test Accuracy: <br/>
> Attention Dropout: 0.6704 <br/>
> Dropout: 0.5759 

For more results please see the notebook, `IMBD-BERT Attention Dropout Analysis.ipynb`.

# Analysis



# Considerations 

The general idea of the technique is very easy to implement and can be applied across many different areas including computer vision, skip-connections, layer-to-layer attention and more. Though only dropout attention is investigated in this project, other variations may be investigated in future research.

# References

[1] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011). 

[2] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, pages 6000–6010.

[3] Srivastava, Nitish, et al. "Dropout: a simple way to prevent neural networks from overfitting." The Journal of Machine Learning Research 15.1 (2014): 1929-1958.
