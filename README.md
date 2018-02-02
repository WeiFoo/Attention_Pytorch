# Notes

This repo is to implement(ish) Bahdanau et al. attention 
paper[ Neural Machine 
Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).

The basic idea of attention is: when you translate a word, you have to use
the most related context information from the source (sentences). In other
words, find the most similar(cosine similarity, etc..) source according to
target. With the attention info, we can better translate words by using
encoder-decoder.


For a toy data set(train=10, test=1), I got the following performance
(testing... it's only one random test data)

```angular2html
output:
epoch:1, train_loss:8.179, test_loss:236.684
epoch:2, train_loss:5.265, test_loss:115.265
epoch:3, train_loss:2.871, test_loss:171.435
epoch:4, train_loss:1.415, test_loss:54.966
epoch:5, train_loss:0.6, test_loss:152.126
epoch:6, train_loss:0.257, test_loss:123.877
epoch:7, train_loss:0.134, test_loss:112.532
epoch:8, train_loss:0.087, test_loss:90.147
epoch:9, train_loss:0.064, test_loss:250.825
epoch:10, train_loss:0.05, test_loss:206.535
epoch:11, train_loss:0.041, test_loss:94.732
epoch:12, train_loss:0.035, test_loss:112.189
epoch:13, train_loss:0.031, test_loss:150.928
epoch:14, train_loss:0.027, test_loss:87.121
epoch:15, train_loss:0.024, test_loss:211.864
epoch:16, train_loss:0.022, test_loss:50.25
epoch:17, train_loss:0.02, test_loss:84.277
epoch:18, train_loss:0.019, test_loss:127.005
epoch:19, train_loss:0.017, test_loss:144.699
epoch:20, train_loss:0.016, test_loss:144.73
epoch:21, train_loss:0.015, test_loss:112.445
epoch:22, train_loss:0.014, test_loss:152.875
epoch:23, train_loss:0.013, test_loss:129.489
epoch:24, train_loss:0.013, test_loss:93.044
epoch:25, train_loss:0.012, test_loss:234.244
epoch:26, train_loss:0.011, test_loss:89.763
epoch:27, train_loss:0.011, test_loss:140.503
....

```

# Todo
1. mini-batch
2. visualize the attention distribution
3. beamsearch

