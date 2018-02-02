# coding: utf-8
"""


"""
import pdb
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

USE_CUDA = torch.cuda.is_available()
MAX_LEN = 100


en_vocab_src = "./Data/vocab.en.txt"
vi_vocab_src = "./Data/vocab.vi.txt"
train_en_src = "./Data/valid.en.txt"
train_vi_src = "./Data/valid.vi.txt"
valid_en_src = "./Data/valid.en.txt"
valid_vi_src = "./Data/valid.vi.txt"
test_en_src = "./Data/test10.en.txt"
test_vi_src = "./Data/test10.vi.txt"


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi = bidirectional
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers,
                          bidirectional=self.bi)

    def forward(self, word_inputs, hidden=None):
        word_inputs = word_inputs.cuda() if USE_CUDA else word_inputs
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        direction = 2 if self.bi else 1
        result = Variable(
            torch.zeros(self.num_layers * direction, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.atten = nn.Linear(2 * hidden_size, hidden_size)
        self.w = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, encoder_outputs, hidden):
        encoder_outputs = encoder_outputs.cuda() if USE_CUDA else encoder_outputs
        source_len = len(encoder_outputs)
        atten_all = Variable(torch.zeros(source_len))
        if USE_CUDA:
            atten_all = atten_all.cuda()

        for i in range(source_len):
            atten_all[i] = self.score(hidden.unsqueeze(0), encoder_outputs[i])
        return self.softmax(atten_all).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_outputs):
        # linear
        res = self.atten(torch.cat((hidden.view(1, -1), encoder_outputs), 1))
        res = self.w.dot(res)
        # dot
        #res = hidden.dot(encoder_outputs)

        return res


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.atten = Attention(hidden_size)

    def forward(self, input, last_context, last_hidden, encoder_outputs):
        input = input.cuda() if USE_CUDA else input
        embedded = self.embedding(input).view(1, 1, -1)

        rnn_input = torch.cat((embedded, last_context.unsqueeze(0)),
                              2)  # combine embedding and last context
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        weights = self.atten(encoder_outputs, hidden)  # get weights
        context = weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)  # 1 x B x N -> B x N
        context = context.squeeze(1)  # B x 1 x N -> B x N

        output = self.softmax(self.out(torch.cat((rnn_output, context), 1)))
        return output, context, hidden, weights

    def initHidden(self):
        result = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result

def read_vocab(src):
    """
    Map word to index as well as index to word.

    :param src: str, src of the word vocabulary
    :return: tuple, (word2index, index2word)

    """
    word2idx = {}
    idx2word = {}
    for i, w in enumerate(open(src).read().splitlines()):
        if w not in word2idx:
            word2idx[w] = i
            idx2word[i] = w
    return word2idx, idx2word


def data_iterator(s_src, t_src, s_vocab, t_vocab, max_sent_len=MAX_LEN,
                  batch_size=1, num_sample=0):
    """
    A data iterator to supply data into model

    :param s_src: str, src of the source data
    :param t_src: str, src of the target data
    :param s_vocab: str, src of the source vocabulary
    :param t_vocab: str, src of the target vocabulary
    :param max_sent_len: int, maximum length of the sentence
    :param batch_size: int(default: 1), batch size
    :param num_sample: int(default: 0),  how many number of data used(good,
                                         for debugging)
    :return:
        yield(source_data, target_data, len_source, len_target)
        source_data: [batch_size, variable]
        target_data: [batch_size, variable]
    """
    s_data = open(s_src, "r").readlines()
    t_data = open(t_src, "r").readlines()
    if num_sample:
        idx = random.sample(range(len(s_data)), num_sample)
        s_data = np.array(s_data)[idx]
        t_data = np.array(t_data)[idx]

    f = lambda x: Variable(torch.LongTensor(x).view(1, -1))
    out_source, out_target, len_source, len_target = [], [], [], []
    batch_idx = 0
    for i, (s_line, t_line) in enumerate(zip(s_data, t_data)):
        if i - batch_idx >= batch_size:
            yield out_source, out_target, len_source, len_target
            out_source, out_target, len_source, len_target = [], [], [], []
            batch_idx = i
        # get the word in vocab
        a_source = [s_vocab[w] if w in s_vocab else s_vocab["<unk>"] for w in
                    s_line.replace("\n", "").split(" ")][:max_sent_len]
        a_target = [t_vocab[w] if w in t_vocab else t_vocab["<unk>"] for w in
                    t_line.replace("/n", "</s>").split()]
        a_target.insert(0, t_vocab["<s>"])
        var_source = f(a_source).cuda() if USE_CUDA else f(a_source)
        var_target = f(a_target).cuda() if USE_CUDA else f(a_target)
        out_source.append(var_source)
        out_target.append(var_target)
        if (i + 1) % batch_size == 0:
            yield (out_source), (out_target), len_source, len_target


def train_one(encoder, decoder, source_vocab, target_vocab, criterion,
              encoder_opt, decoder_opt, atten_opt):
    """
     Train one epoch.

    :param encoder: EncoderRNN, the encoder model
    :param decoder: DecoderRNN, the decoder model
    :param source_vocab: dict, the source vocabulary
    :param target_vocab: dict, the target vocabulary
    :param criterion: loss function
    :param encoder_opt: encoder optimizer
    :param decoder_opt: decoder optimizer
    :param atten_opt: attention optimizer
    :return:
            training loss of this batch
    """

    total_len = 0
    total_loss = 0
    data = data_iterator(train_en_src, train_vi_src, source_vocab,target_vocab)

    for source, target, _, _ in data:  ## TODO: batch
        for s, t in zip(source, target):
            s = s.view(-1)
            t = t.view(-1)
            target_len = t.size()[0]

            encoder_hidden = encoder.initHidden()
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            encoder_outputs, encoder_hidden = encoder(s, encoder_hidden)
            decode_hidden = encoder_hidden
            decode_context = Variable(torch.zeros(1, decoder.hidden_size))
            decode_context = decode_context.cuda() if USE_CUDA else decode_context
            loss = 0

            for de_i in range(target_len):
                out = decoder(t[de_i],decode_context,decode_hidden,encoder_outputs)
                decode_out, context, hidden = out[0], out[1],out[2]
                loss += criterion(decode_out, t[de_i])
                decode_context = context
                decode_hidden = hidden

            loss.backward()
            encoder_opt.step()
            decoder_opt.step()
            atten_opt.step()

            total_len += target_len
            total_loss += loss.data[0]

    return 0 if total_len == 0 else total_loss / total_len


def evaluate(encoder, decoder, en_sentence, vi_sentence):
    """
    Evaluate how the current model perform while training.

    :param encoder: EncoderRNN, the encoder model
    :param decoder: DecoderRNN, the decoder model
    :param en_sentence: list, list of source words in one sentence
    :param vi_sentence: list, list of target words in one sentence
    :return:
        loss: float, the loss.

    """
    # Run through encoder
    encoder_hidden = encoder.initHidden()
    encoder_outputs, encoder_hidden = encoder(en_sentence.view(-1),encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = vi_sentence.view(-1)
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input
    decoder_context = decoder_context.cuda() if USE_CUDA else decoder_context

    decoder_hidden = encoder_hidden.clone().view(1, 1, -1)
    criterion = nn.NLLLoss()
    loss = 0
    # Run through decoder
    for di in range(decoder_input.size()[0]):
        out = decoder(decoder_input[di], decoder_context,
                      decoder_hidden, encoder_outputs)
        decoder_output = out[0]
        decoder_context = out[1]
        decoder_hidden = out[2]
        predict = decoder_output.view(1, -1)
        actual = decoder_input[di].view(-1)
        loss += criterion(predict, actual).data[0]

    return loss


def train(encoder, decoder, source_vocab, target_vocab, n_epoches=200,
          learning_rate=0.01, print_every_ep=1):
    """
    Train the model for n_epoches

    :param encoder: EncoderRNN, the encoder model
    :param decoder: DecoderRNN, the decoder model
    :param source_vocab: dict, the source vocabulary
    :param target_vocab: dict, the target vocabulary
    :param n_epoches: int, number of epoches
    :param learning_rate: float, learning rates
    :param print_every_ep: int, print every epoches.
    :return: None.
    """
    encoder_opt = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_opt = optim.SGD(decoder.parameters(), lr=learning_rate)
    atten_opt = optim.SGD(decoder.atten.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    total_loss = 0
    print_loss = 0

    for ep in range(1, n_epoches + 1):
        loss = train_one(encoder, decoder, source_vocab, target_vocab,
                         criterion, encoder_opt, decoder_opt, atten_opt)
        total_loss += loss
        print_loss += loss

        if ep % print_every_ep == 0:
            loss_avg = print_loss / print_every_ep
            print_loss = 0
            data = data_iterator(test_en_src, test_vi_src, source_vocab,
                                 target_vocab)
            total_test_loss = 0
            num_test = 0
            for test_en_batch, test_vi_batch, _, _ in data:  ## TODO: batch
                for test_en, test_vi in zip(test_en_batch, test_vi_batch):
                    test_en = test_en.view(1, 1, -1)
                    test_vi = test_vi.view(1, 1, -1)
                    test_loss = evaluate(encoder, decoder, test_en, test_vi)
                    total_test_loss += test_loss
                    num_test += 1
            test_loss = total_test_loss/num_test
            print("epoch:{}, train_loss:{}, test_loss:{}".format(ep, round(
                loss_avg, 3), round(test_loss, 3)))

def run():
    hidden_size = 256
    source_vocab, idx2source = read_vocab(en_vocab_src)
    target_vocab, idx2target = read_vocab(vi_vocab_src)
    encoder = EncoderRNN(len(source_vocab), hidden_size)
    encoder = encoder.cuda() if USE_CUDA else encoder
    decoder = DecoderRNN(hidden_size, len(target_vocab))
    decoder = decoder.cuda() if USE_CUDA else decoder
    train(encoder, decoder, source_vocab, target_vocab)


if __name__ == "__main__":
    run()
