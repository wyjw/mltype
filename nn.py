import torch
import numpy as np
import pickle as pickle
from sklearn.svm import SVC
from sklearn.svm import LinearSVC, NuSVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from gensim.summarization.textcleaner import split_sentences, tokenize_by_word
from gensim.utils import simple_preprocess
from LSTM_classifier import LSTMClassifier
import time
from torchtext import data
from torchtext import datasets
import torchtext
import random
from CNN_classifier import CNN1d
from f1_score import precision_recall, F_score, print_f_s


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_preds(model, iterator):
    model.eval()
    ret = []
    real = []
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            predictions = predictions.cpu()
            for (i,x) in enumerate(batch):
                #ret.append([batch.text[i], predictions[i]])
                ret.append(predictions[i])
                real.append(batch.label[i].cpu())
    return ret, real


class TypePredDataset(torchtext.data.Dataset):
    def __init__(self, text_field, label_field, mooc_reader, **kwargs):
        fields = [("text",text_field), ("label", label_field)]
        examples = []
        for (i, x) in enumerate(mooc_reader.z): #TODO
            examples.append(data.Example.fromlist([x[0], x[1]], fields))
        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex) : return len(ex.text)

    @classmethod
    def splits(cls, text_field, label_field, root='.data', train='train', test='test', **kwargs):
        return super().splits(root, text_field=text-field, label_field=label_field, train=train, validation=None, test=test, **kwargs)


if __name__ == "__main__":
    SEED = 1234
    torch.manual_seed(SEED)
    torch.backedns.cudnn.deterministic = True

    BATCH_SIZE = 32

    TEXT = data.Field(sequential = True, fix_length = 32)
    LABEL = data.LabelField(sequential = True, dtype = torch.long)
    
    pred_dataset = TypePredDataset(TEXT, LABEL)
    train_data, test_data = preq.split()
    train_data, valid_data = train_data.split(random_state = random.seed(SEED))

    print(f'Test data has {len(test_data)} samples')
    print(f'Valid data has {len(valid_data)} samples')
    print(f'Train data has {len(train_data)} samples')
    MAX_VOCAB_SIZE = 25000

    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size = BATCH_SIZE,
            device = device)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [2,3,4]
    OUTPUT_DIM = len(LABEL.vocab)
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = CNN1d(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Declares the padded character and non-chars
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    import torch.optim as optim
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 100

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut4-model.pt')
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load('tut4-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print('Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

    x, y = get_preds(model, test_iterator)
    print(f'{len(x)} and {len(y)} is the length of x and y')

    non_zero = 0
    for i in x:
        print("For example got solution", i)
        if (i.argmax() != 0):
            non_zero += 1
    print("Predictions are ", non_zero, "nonzero out of total ", len(x))
    maxes = []
    for i in x:
        maxes.append(i.argmax())

    precision, recall, TP, TP_plus_FN, TP_plus_FP = precision_recall(np.array(maxes).tolist(), np.array(y).tolist())
    print('precision')
    print(precision)
    print('recall')
    print(recall)
    print('TP')
    print(TP)
    print('TP_plus_FN')
    print(TP_plus_FN)
    print('TP_plus_FP')
    print(TP_plus_FP)
    # print(dic)


    f_scores = F_score(precision, recall)
    print('f_scores')
    print(f_scores)
    # print(f_scores.keys())


    print('\r')
    print_f_score(np.array(maxes).tolist(), np.array(y).tolist())
