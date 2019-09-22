from util import *
from model import LSTMSequence
from torch.utils.data import DataLoader
import torch.optim as optim


if __name__ == "__main__":
    #### Load vua data set ####
    raw_train, raw_validation, raw_test, vocabulary = load_data_set("vua",vocabulary_mode=1)
    # generate word2idx and idx2word mappings
    word2idx, idx2word = generate_w2idx_idx2w(vocabulary)
    # generate in-memory embedding matrix
    embedding_matrix = get_embedding_matrix(word2idx, idx2word)

    # embed train and validation
    # sample: [sentence, labels]
    # dataset = [sample,sample,....]
    embedded_train = [[embed_sequence(sample[0], word2idx,embedding_matrix), sample[1]] for sample in raw_train]
    embedded_val = [[embed_sequence(sample[0], word2idx,embedding_matrix), sample[1]] for sample in raw_validation]


    # Separate the input (embedded_sequence) and labels in the indexed train sets.
    # embedded_train_vua: embedded_sentence, pos, labels
    train_dataset = TextDataset([example[0] for example in embedded_train],
                                    [example[1] for example in embedded_train])

    val_dataset = TextDataset([example[0] for example in embedded_val],
                                  [example[1] for example in embedded_val])

    # Set data loader with 64-samples per batch.
    batch_size = 64


    train_dataloader= DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=TextDataset.collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                    collate_fn=TextDataset.collate_fn)

    # Instantiate model
    model = LSTMSequence(num_classes=2, embedding_dim=300, hidden_size=300, num_layers=1, bidir=True,
                                dropout1=0.5, dropout2=0, dropout3=0.1)

    # Set up criterion for calculating loss
    loss_criterion = nn.NLLLoss()
    rnn_optimizer = optim.Adam(model.parameters(), lr=0.005)
    # Number of epochs (passes through the dataset) to train the model for.
    num_epochs = 10

    train_loss = []
    val_loss = []
    performance_matrix = None
    val_f1 = []
    val_p = []
    val_r = []
    val_acc = []
    train_f1 = []
    # A counter for the number of network gradient updates
    num_iter = 0
    comparable = []
    for epoch in range(num_epochs):
        print("Starting epoch Number {}".format(epoch + 1))
        for (example_text, example_lengths, labels) in train_dataloader:
            example_text = Variable(example_text)
            example_lengths = Variable(example_lengths)
            labels = Variable(labels)
            # predicted is with shape: (batch_size, seq_len, 2)
            predicted = model(example_text, example_lengths)
            batch_loss = loss_criterion(predicted.view(-1, 2), labels.view(-1))
            rnn_optimizer.zero_grad()
            batch_loss.backward()
            rnn_optimizer.step()
            num_iter += 1
            # Calculate validation and training set loss and accuracy every 50 network gradient updates
            if num_iter % 50 == 0:
                model.eval()
                # total_examples = total number of words
                total_examples = 0
                total_eval_loss = 0
                for (eval_text, eval_lengths, eval_labels) in val_dataloader:
                    with torch.no_grad():
                        eval_text = Variable(eval_text)
                        eval_lengths = Variable(eval_lengths)
                        eval_labels = Variable(eval_labels)


                    predicted = model(eval_text, eval_lengths)
                    total_eval_loss += loss_criterion(predicted.view(-1, 2), eval_labels.view(-1))
                    # get 0 , 1 predictions
                    _, predicted_labels = torch.max(predicted.data, 2)
                    total_examples += eval_lengths.size(0)
                    average_eval_loss = total_eval_loss / val_dataloader.__len__()

                print("Iteration Number {}. Percent Validation Loss {}.".format(num_iter, average_eval_loss))

    print("**********************************************************")
    print("Evalutation on test set: ")
    print("**********************************************************")

    print('number of samples for test_set ', len(raw_test))

    embedded_test_vua = [[embed_sequence(sample[0], word2idx, embedding_matrix), sample[1]] for sample in raw_test]

    # Separate the input
    test_dataset = TextDataset([example[0] for example in embedded_test_vua],
                               [example[1] for example in embedded_test_vua])


    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 collate_fn=TextDataset.collate_fn)

    # Set model to eval mode to turn off dropout.
    model.eval()

    predictions = []
    for (eval_text, eval_lengths, eval_labels) in test_dataloader:
        with torch.no_grad():
            eval_text = Variable(eval_text)
            eval_lengths = Variable(eval_lengths)
            eval_labels = Variable(eval_labels)

        predicted = model(eval_text, eval_lengths)
        # get 0 , 1 predictions
        # predicted_labels: (batch_size, seq_len)
        _, predicted_labels = torch.max(predicted.data, 2)

        pred_lst = []
        total = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(eval_labels.size()[0]):  # each example i.e. each row
            for j in range(eval_labels.data[i].size()[0]):
                eval_val = eval_labels.data[i][j].item()
                prediction_val = predicted_labels.data[i][j].item()
                if eval_val == prediction_val:
                    if eval_val == 0:
                        tn += 1
                    else:
                        tp += 1
                else:
                    if eval_val == 1:
                        fn += 1
                    else:
                        fp += 1

    accuracy = (tp + tn)/(tn+fp+fn+tp)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = (2 * precision * recall) / (precision +recall)
    print("Results: \n")
    print("Accuracy: %s \n" %str(100*accuracy))
    print("Precision: %s \n" % str(100*precision))
    print("Recall: %s \n" % str(100*recall))
    print("F1-Score: %s \n" % str(100*f1))

