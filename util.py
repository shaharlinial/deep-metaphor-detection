import csv, ast, torch,  torch.nn as nn, numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import Dataset

# input: relative dataset filepath
# output: raw train, raw validation, vocabulary set
def load_data_set(name, vocabulary_mode=1):
    # vocabulary mode:
    #   1 : train only
    #   2 : train and validation
    raw_val = []
    raw_train = []
    raw_test = []
    vocabulary = []
    # default is VUA
    # path for training and validation sets
    train_path = "datasets/VUAsequence/VUA_seq_formatted_train.csv"
    validation_path = "datasets/VUAsequence/VUA_seq_formatted_val.csv"
    test_path = "datasets/VUAsequence/VUA_seq_formatted_test.csv"
    # column index for sentence,label in csv
    sentence_index = 2
    label_index = 3

    if name == "vua":
        train_path = "datasets/VUAsequence/VUA_seq_formatted_train.csv"
        validation_path = "datasets/VUAsequence/VUA_seq_formatted_val.csv"
        test_path = "datasets/VUAsequence/VUA_seq_formatted_test.csv"

        sentence_index = 2
        label_index = 3

    with open(train_path, encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            label_seq = ast.literal_eval(line[label_index])
            raw_train.append([line[sentence_index], label_seq])
            vocabulary.extend(line[sentence_index].split())

    with open(validation_path, encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            label_seq = ast.literal_eval(line[label_index])
            raw_val.append([line[sentence_index], label_seq])
            if vocabulary_mode == 2:
                vocabulary.extend(line[sentence_index].split())

    with open(test_path, encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            label_seq = ast.literal_eval(line[label_index])
            raw_test.append([line[sentence_index], label_seq])
            if vocabulary_mode == 2:
                vocabulary.extend(line[sentence_index].split())

    return raw_val, raw_train, raw_test, set(vocabulary)

# input: a bag of words
# output: word2index and index2word dictionary mapping
def generate_w2idx_idx2w(vocabulary):
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    for word in vocabulary:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word

#input: word2idx and idx2word mapping
#ouput: nn.Embedding matrix
def get_embedding_matrix(word2idx, idx2word, normalization=False):
    # Load the GloVe vectors into a dictionary, keeping only words in vocabulary
    embedding_dim = 300
    glove_path = "glove/glove840B300d.txt"
    glove_vectors = {}
    with open(glove_path) as glove_file:
        for line in tqdm(glove_file, total=sum(1 for line in open(glove_path))):
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
    # Calculate mean and stdev of word embeddings
    all_embeddings = np.array(list(glove_vectors.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_stdev = float(np.std(all_embeddings))
    print("Embeddings mean: ", embeddings_mean)
    print("Embeddings stdev: ", embeddings_stdev)

    vocab_size = len(word2idx)
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean, embeddings_stdev)
    # Go through the embedding matrix and replace the random vector with a
    # embedded one if it is available.
    # Start iteration at 2 since 0, 1 are PAD, UNK
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

def embed_sequence(sentence, word2idx, glove_embeddings):

    words = sentence.split()

    # 1. embed the sequence by glove vector
    # Replace words with tokens, and 1 (UNK index) if words not indexed.
    indexed_sequence = [word2idx.get(x, 1) for x in words]
    # glove_part has shape: (seq_len, glove_dim)
    glove_embedded = glove_embeddings(Variable(torch.LongTensor(indexed_sequence)))

    return glove_embedded.data

class TextDataset(Dataset):
    def __init__(self, embedded_text, labels):
        if len(embedded_text) != len(labels):
            raise ValueError("Differing number of sentences and labels!")
        # A list of numpy arrays, where each inner numpy arrays is sequence_length * embed_dim
        # embedding for each word is : glove
        self.embedded_text = embedded_text
        #  a list of list: each inner list is a sequence of 0, 1.
        # where each inner list is the label for the sentence at the corresponding index.
        self.labels = labels

    def __getitem__(self, idx):
        example_text = self.embedded_text[idx]
        example_label_seq = self.labels[idx]
        # Truncate the sequence if necessary
        example_length = example_text.shape[0]
        assert (example_length == len(example_label_seq))
        return example_text, example_length, example_label_seq

    def __len__(self):
        """
        Return the number of examples in the Dataset.
        """
        return len(self.labels)

    @staticmethod
    def collate_fn(batch):
        batch_padded_example_text = []
        batch_lengths = []
        batch_padded_labels = []

        # Get the length of the longest sequence in the batch
        max_length = -1
        for text, length, labels in batch:
            if length > max_length:
                max_length = length

        # Iterate over each example in the batch
        for text, length, label in batch:
            # Unpack the example (returned from __getitem__)
            # Amount to pad is length of longest example - length of this example.
            amount_to_pad = max_length - length
            # Tensor of shape (amount_to_pad,), converted to LongTensor
            pad_tensor = torch.zeros(amount_to_pad, text.shape[1])

            # Append the pad_tensor to the example_text tensor.
            # Shape of padded_example_text: (padded_length, embedding_dim)
            # top part is the original text numpy,
            # and the bottom part is the 0 padded tensors

            # text from the batch is a np array, but cat requires the argument to be the same type
            # turn the text into a torch.FloatTenser, which is the same type as pad_tensor
            text = torch.Tensor(text)
            padded_example_text = torch.cat((text, pad_tensor), dim=0)

            # pad the labels with zero.
            padded_example_label = label + [0] * amount_to_pad

            # Add the padded example to our batch
            batch_padded_example_text.append(padded_example_text)
            batch_lengths.append(length)
            batch_padded_labels.append(padded_example_label)

        # Stack the list of LongTensors into a single LongTensor
        return (
                torch.stack(batch_padded_example_text),
                torch.LongTensor(batch_lengths),
                torch.LongTensor(batch_padded_labels))