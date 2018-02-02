# Notes

This repo is to implement(ish) Bahdanau et al. attention 
paper[ Neural Machine 
Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).

The basic idea of attention is: when you translate a word, you have to use
the most related context information from the source (sentences). In other
words, find the most similar(cosine similarity, etc..) source according to
target. With the attention info, we can better translate words by using
encoder-decoder.


For a toy data set(train=10, test=10), I got the following performance


```angular2html
output:
epoch:1, train_loss:8.162, test_loss:236.723
epoch:2, train_loss:5.12, test_loss:209.985
epoch:3, train_loss:2.695, test_loss:203.862
epoch:4, train_loss:1.351, test_loss:198.519
epoch:5, train_loss:0.592, test_loss:195.689
epoch:6, train_loss:0.261, test_loss:195.262
epoch:7, train_loss:0.135, test_loss:195.599
epoch:8, train_loss:0.087, test_loss:195.848
epoch:9, train_loss:0.063, test_loss:196.005
epoch:10, train_loss:0.05, test_loss:196.145
epoch:11, train_loss:0.041, test_loss:196.278
epoch:12, train_loss:0.035, test_loss:196.406
epoch:13, train_loss:0.031, test_loss:196.527
epoch:14, train_loss:0.027, test_loss:196.643
epoch:15, train_loss:0.024, test_loss:196.754
epoch:16, train_loss:0.022, test_loss:196.86
epoch:17, train_loss:0.02, test_loss:196.961
epoch:18, train_loss:0.018, test_loss:197.058
epoch:19, train_loss:0.017, test_loss:197.15
epoch:20, train_loss:0.016, test_loss:197.239


```

# Todo
1. mini-batch
2. visualize the attention distribution
3. beamsearch

