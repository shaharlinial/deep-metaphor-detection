import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import math
import mmap
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import ast
import sklearn.metrics as metrics
import numpy as np
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
using_GPU = torch.cuda.is_available()

print("GPU Detected: %s" %(using_GPU))

SOS_token = 0
vocab = set()
#word_to_ix = {"<PAD>": 0, "<UNK>": 1}
#idx_to_word = {0: "<PAD>", 1: "<UNK>"}

ix_to_label = {0: "Literal", 1: "Metaphor"}

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def get_word2idx_idx2word(vocab):
    """

    :param vocab: a set of strings: vocabulary
    :return: word2idx: string to an int
             idx2word: int to a string
    """
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    for word in vocab:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word

def embed_indexed_sequence(sentence, word2idx, glove_embeddings):
    """
    Assume that pos_seq maps well with sentence
    Assume that the given sentence is indexed by word2idx
    Assume that word2idx has 1 mapped to UNK
    Assume that word2idx maps well implicitly with glove_embeddings
    Assume that the given pos_seq is indexed by pos2idx
    Assume that pos2idx maps well implicitly with pos_embeddings
    i.e. the idx for each word is the row number for its corresponding embedding

    :param sentence: a single string: a sentence with space
    :param pos_seq: a list of ints: indexed pos_sequence
    :param word2idx: a dictionary: string --> int
    :param glove_embeddings: a nn.Embedding with padding idx 0
    :param elmo_embeddings: a h5py file
                    each group_key is a string: a sentence
                    each inside group is an np array (seq_len, 1024 elmo)
    :param pos_embeddings: a nn.Embedding without padding idx
    :return: a np.array (seq_len, embed_dim=glove+elmo+suffix)
    """
    words = sentence

    # 1. embed the sequence by glove vector
    # Replace words with tokens, and 1 (UNK index) if words not indexed.
    indexed_sequence = [word2idx.get(x, 1) for x in words]
    # glove_part has shape: (seq_len, glove_dim)
    glove_part = glove_embeddings(Variable(torch.LongTensor(indexed_sequence)))

    return glove_part.data


def get_embedding_matrix(word2idx, idx2word, normalization=False):
    """
    assume padding index is 0

    :param word2idx: a dictionary: string --> int, includes <PAD> and <UNK>
    :param idx2word: a dictionary: int --> string, includes <PAD> and <UNK>
    :param normalization:
    :return: an embedding matrix: a nn.Embeddings
    """
    # Load the GloVe vectors into a dictionary, keeping only words in vocab
    embedding_dim = 300
    glove_path = "glove/glove840B300d.txt"
    glove_vectors = {}
    with open(glove_path) as glove_file:
        for line in tqdm(glove_file, total=get_num_lines(glove_path)):
            split_line = line.rstrip().split()
            word = split_line[0]
            if len(split_line) != (embedding_dim + 1) or word not in word2idx:
                continue
            assert (len(split_line) == embedding_dim + 1)
            vector = np.array([float(x) for x in split_line[1:]], dtype="float32")
            if normalization:
                vector = vector / np.linalg.norm(vector)
            assert len(vector) == embedding_dim
            glove_vectors[word] = vector

    print("Number of pre-trained word vectors loaded: ", len(glove_vectors))

    # Calculate mean and stdev of embeddings
    all_embeddings = np.array(list(glove_vectors.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_stdev = float(np.std(all_embeddings))
    print("Embeddings mean: ", embeddings_mean)
    print("Embeddings stdev: ", embeddings_stdev)

    # Randomly initialize an embedding matrix of (vocab_size, embedding_dim) shape
    # with a similar distribution as the pretrained embeddings for words in vocab.
    vocab_size = len(word_to_ix)
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean, embeddings_stdev)
    # Go through the embedding matrix and replace the random vector with a
    # pretrained one if available. Start iteration at 2 since 0, 1 are PAD, UNK
    for i in range(2, vocab_size):
        word = idx2word[i]
        if word in glove_vectors:
            embedding_matrix[i] = torch.FloatTensor(glove_vectors[word])
    if normalization:
        for i in range(vocab_size):
            embedding_matrix[i] = embedding_matrix[i] / float(np.linalg.norm(embedding_matrix[i]))
    embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    embeddings.weight = nn.Parameter(embedding_matrix)
    return embeddings



# input size is the ONE_HOT_ENCODING vector size that is passed to the Embedding layer to create a more dense encoding
# i.e it is the vocabulary size
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # TODO: Replace the embeddings with pre-trained word embeddings such as word2vec or GloVe
        #self.embedding = nn.Embedding(input_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        #embedded = self.embedding(input).view(1, 1, -1)
        #output = embedded

        output, hidden = self.lstm(input.view(1,1,-1), hidden)
        return output, hidden

    # (hidden_state, cell_state)
    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device), torch.zeros(1, 1, self.hidden_size, device=device))

# The final output size is 2 - (p_literal, p_metaphor)
# maximum sentence length (input length, for encoder outputs) that it can apply to.
# Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    # (hidden_state, cell_state)
    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device), torch.zeros(1, 1, self.hidden_size, device=device))


def prepareVUAtrainData():
    training_pairs = []
    max_length = 0
    with open('datasets/VUAsequence/VUA_seq_formatted_train.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            sentence = line[2].split()
            label_seq = ast.literal_eval(line[3])
            training_pairs.append((sentence, label_seq))
            if len(sentence) > max_length:
                max_length = len(sentence)

    for sentence, tags in training_pairs:
        for word in sentence:
            vocab.add(word)

    return training_pairs, max_length


def prepareVUAtestData():
    test_pairs = []
    max_length = 0
    with open('datasets/VUAsequence/VUA_seq_formatted_test.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            sentence = line[2].split()
            label_seq = ast.literal_eval(line[3])
            test_pairs.append((sentence, label_seq))
            if len(sentence) > max_length:
                max_length = len(sentence)

    for sentence, tags in test_pairs:
        for word in sentence:
            vocab.add(word)
    return test_pairs, max_length

# To train, for each pair we will need an input tensor (indexes of the words in the input sentence) and target tensor
# with metaphor indicators in each position


def indexesFromSentence(word_to_ix, sentence):
    return [word_to_ix[word] for word in sentence]


def tensorFromTags(tags):
    return torch.tensor(tags, dtype=torch.long, device=device).view(-1, 1)


def tensorFromSentence(word_to_ix, sentence):
    indexes = indexesFromSentence(word_to_ix, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair( pair):
    input_tensor = pair[0]
    target_tensor = tensorFromTags(pair[1])
    return input_tensor, target_tensor


# This is a helper function to print time elapsed and estimated time remaining given the current time and progress %.

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

## We call train for each pass of a sequence (sentence) through the model


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # pass every word through the encoder and save all the outputs
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)  # top value, top index
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, word_to_ix, pairs, n_epochs, max_length, print_every=1000, plot_every=100, learning_rate=0.02):
    start = time.time()
    print_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(pair) for pair in pairs]
    criterion = nn.NLLLoss()

    for epoch in range(n_epochs):
        for iter in range(1, len(training_pairs) + 1):
            training_pair = training_pairs[iter - 1]

            input_tensor = Variable(training_pair[0])
            target_tensor = Variable(training_pair[1])
            if using_GPU:
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)

            print_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / len(training_pairs)),
                                             iter, iter / len(training_pairs) * 100, print_loss_avg))
        print("Finished Epoch Number : %s" % (str(epoch)))
    print("finished training")


def evaluate(test_sentences, encoder, attn_decoder, max_length):
    c_matrix = np.zeros((2, 2))

    for pair in test_sentences:
        sentence = pair[0]
        labels = pair[1]
        predicted_labels = predictSentenceLabels(sentence, encoder, attn_decoder, max_length)
        c_matrix = np.add(c_matrix, metrics.confusion_matrix(labels, predicted_labels, [0, 1]))

    tn, fp, fn, tp = c_matrix.ravel()
    tn = tn or 0.0000001
    fp = fp or 0.0000001
    fn = fn or 0.0000001
    tp = tp or 0.0000001
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("Accuracy: %s \n Precision: %s \n Recall: %s \n F1-Score: %s" %(str(accuracy),str(precision),str(recall),str(f1)))

def predictSentenceLabels(sentence, encoder, decoder, max_length):
    with torch.no_grad():
        input_tensor = sentence
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        decoded_labels = []

        for di in range(input_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)  # top value, top index
            pred_label = topi.item()
            decoded_labels.append(pred_label)

            decoder_input = topi.squeeze().detach()

        return decoded_labels



train_sentences, max_train_sentence_len = prepareVUAtrainData()
test_sentences, max_test_sentence_len = prepareVUAtestData()
word_to_ix,idx_to_word = get_word2idx_idx2word(vocab)
glove_embeddings = get_embedding_matrix(word_to_ix,idx_to_word)

embedded_train = [(embed_indexed_sequence(sen[0], word_to_ix, glove_embeddings),sen[1]) for sen in train_sentences]
embedded_test = [(embed_indexed_sequence(sen[0], word_to_ix, glove_embeddings),sen[1]) for sen in test_sentences]


max_length = max(max_train_sentence_len, max_test_sentence_len)
hidden_size = 300
vocab_size = len(word_to_ix)
n_epochs = 15
encoder = EncoderRNN(hidden_size).to(device)
if using_GPU:
    encoder.cuda()
output_feature_size = len(ix_to_label)  # 2 : we have two classes - literal/metaphor

attn_decoder = AttnDecoderRNN(hidden_size, output_feature_size, max_length, dropout_p=0.1).to(device)
if using_GPU:
    attn_decoder.cuda()
trainIters(encoder, attn_decoder, word_to_ix, embedded_train, n_epochs, max_length, print_every=5000)

evaluate(embedded_test, encoder, attn_decoder, max_length)

#http://nlp.stanford.edu/data/glove.840B.300d.zip