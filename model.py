import torch.nn as nn
import torch.nn.functional as F
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class LSTMSequence(nn.Module):
    def __init__(self, num_classes, embedding_dim, hidden_size, num_layers, bidir=True,
                 dropout1=0.2, dropout2=0.2, dropout3=0.2):
        #  call the superclass (nn.Module) constructor first
        super(LSTMSequence, self).__init__()

        # Set up the RNN: use an bi-LSTM here.
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout2, batch_first=True, bidirectional=bidir)
        direc = 2 if bidir else 1

        # Set up the final transform to a distribution over classes.
        self.output_projection = nn.Linear(hidden_size * direc, num_classes)

        # Dropout layer
        self.dropout_on_input_to_LSTM = nn.Dropout(dropout1)
        self.dropout_on_input_to_linear_layer = nn.Dropout(dropout3)


    def forward(self, inputs, lengths):
        # run LSTM
        # apply dropout to the input
        # Shape of inputs: (batch_size, sequence_length, embedding_dim)
        embedded_input = self.dropout_on_input_to_LSTM(inputs)
        # Sort the embedded inputs by decreasing order of input length. [ this is done for batching ]
        # sorted_input shape: (batch_size, sequence_length, embedding_dim)
        (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(embedded_input, lengths)
        # Pack the sorted inputs with pack_padded_sequence.
        packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)
        # Run the input through the RNN.
        packed_sorted_output, _ = self.rnn(packed_input)
        # Unpack (pad) the input with pad_packed_sequence
        # Shape: (batch_size, sequence_length, hidden_size)
        sorted_output, _ = pad_packed_sequence(packed_sorted_output, batch_first=True)
        # Re-sort the packed sequence to restore the initial ordering
        # Shape: (batch_size, sequence_length, hidden_size)
        output = sorted_output[input_unsort_indices]
        # 2. run linear layer
        # apply dropout to input to the linear layer
        # (batch_size, sequence_length, hidden_size)
        input_encoding = self.dropout_on_input_to_linear_layer(output)
        # Run the RNN encoding of the input through the output projection
        # to get scores for each of the classes.
        # (batch_size, sequence_length, 2)
        unnormalized_output = self.output_projection(input_encoding)
        # Normalize with log softmax
        output_distribution = F.log_softmax(unnormalized_output, dim=-1)
        return output_distribution