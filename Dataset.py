import json
import re
import logging
import os.path
import sys
import multiprocessing
import torch

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import gensim
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
from f1_score import precision_recall, F_score, print_f_score

data_dir = os.getcwd()
more_dir = data_dir + '/Dataset/DSA/'
labeledFilename = more_dir + 'DSA_LabeledFile'
jsonFilenameList = [more_dir + 'Captions_algorithms_Princeton.json', more_dir + 'Captions_algorithms_Stanford.json', more_dir + 'Captions_data-structure-and-algorithm_UC-San-Diego.json']
coreConceptsFile = more_dir + 'CoreConcepts_DSA'

class MOOCReader:
    # Class for reading in the file from the dataset
    def __init__(self):
        type = 'DS'
        self.dataset = []
        self.word2idx = {}
        self.idx2word = {}
        self.label2idx = {}
        self.idx2label = {}
        self.read_json(jsonFilenameList)
        self.read_labeled_file(labeledFilename)
        self.read_important_words(coreConceptsFile)
        self.organizeData()
        self.pairUp()
        self.constructTrainingSet()
        self.data_split()

    def read_json(self, filename):
        self.data = []
        for filename in jsonFilenameList:
            with open(filename, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))

    def read_labeled_file(self, filename):
        # Gets the labels for prerequisites
        self.label = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                if len(line.split('\t')) > 3:
                    temp_entry = {}
                    temp_entry['first'] = line.split('\t')[0]
                    temp_entry['second'] = line.split('\t')[2]
                    temp_entry['rel'] = line.split('\t')[4]
                    self.dataset.append(temp_entry)

    def read_important_words(self, filename):
        count = 0
        self.int2concept = {}
        self.concept2int = {}
        self.conceptList = []
        with open(filename, 'r') as f:
            for line in f:
                for word in line.split("::;"):
                    word = word.replace('\n', '')
                    if count in self.int2concept:
                        self.int2concept[count].append(word)
                    else:
                        self.int2concept[count] = [word]
                    self.concept2int[word] = count
                    self.conceptList.append(word)
                count += 1

    def organizeData(self):
        self.byConcept = {}
        for concept in self.conceptList:
            if concept not in self.byConcept:
                self.byConcept[concept] = {}
                self.byConcept[concept]['appears'] = []
        for concept in self.byConcept:
            for (ind, entry) in enumerate(self.data):
                if concept.lower() in entry['text'].lower():
                    if ind not in self.byConcept[concept]['appears']:
                        self.byConcept[concept]['appears'].append(ind)
        # This part is for the extraction of sentences
        for concept in self.byConcept:
            for case in self.dataset:
                if case['first'] == concept:
                    if 'next' not in self.byConcept[concept]:
                        self.byConcept[concept]['next'] = []
                        self.byConcept[concept]['rel'] = []
                    if case['second'] not in self.byConcept[concept]['next']:
                        self.byConcept[concept]['next'].append(case['second'])
                        self.byConcept[concept]['rel'].append(case['rel'])

    def pairUp(self):
        self.pairUp = []
        # Checks if they are in the same document
        for concept in self.byConcept:
            # for storage of file number and information about pairing
            curr_conc = self.byConcept[concept]
            if 'appears' in curr_conc and 'next' in curr_conc:
                first_app = curr_conc['appears']
                for idx, next_entry in enumerate(curr_conc['next']):
                    pair_dict = {}
                    if 'appears' in self.byConcept[next_entry]:
                        second_app = self.byConcept[next_entry]['appears']
                        # intersection where both files exist
                        pair_dict['int'] = list(set(first_app) & set(second_app))
                        # get the relation needed
                        pair_dict['rel'] = curr_conc['rel'][idx]
                        pair_dict['first'] = concept
                        pair_dict['second'] = next_entry
                        self.pairUp.append(pair_dict)

        # Check at the sentence level
        for element in self.pairUp:
            for story_num in element['int']:
                temp_sentences = split_sentences(self.data[story_num]['text'])
                first_sentence = list(filter(lambda x : element['first'] in x, temp_sentences))
                second_sentence = list(filter(lambda x : element['second'] in x, temp_sentences))
                intersect = list(set(first_sentence) & set(second_sentence))
                if len(intersect) > 0:
                    element['joint_sentence'] = tokenize_list_words(intersect)
                else:
                    element['first_sentence'] = tokenize_list_words(first_sentence)
                    element['second_sentence'] = tokenize_list_words(second_sentence)

    def constructTrainingSet(self):
        self.final_data = []
        for element in self.pairUp:
            dict_data = {}
            first_phrase = simple_preprocess(element['first'])
            second_phrase = simple_preprocess(element['second'])
            # This ensures that there is a sentence used by both.
            if 'joint_sentence' in element:
                temp_sent = []
                temp_or_sent = []
                # marks the first term
                temp_first_mark = []
                # marks the second term
                temp_second_mark = []
                for word in element['joint_sentence']:
                    if word not in self.word2idx:
                        curr_id = len(list(self.word2idx))
                        self.word2idx[word] = curr_id
                        self.idx2word[word] = curr_id
                    temp_sent.append(self.word2idx[word])
                    temp_or_sent.append(word)
                for type in element['rel']:
                    if type not in self.label2idx:
                        curr_id = len(list(self.label2idx))
                        self.label2idx[element['rel']] = curr_id
                        self.idx2word[element['rel']] = curr_id
                dict_data['token'] = temp_sent
                dict_data['ortoken'] = temp_or_sent
                dict_data['rel'] = self.label2idx[element['rel']]
                dict_data['orrel'] = element['rel']
                dict_data['find_one'] = sublistfind(first_phrase, element['joint_sentence'])
                dict_data['find_two'] = sublistfind(second_phrase, element['joint_sentence'])
            else:
                temp_sent = []
                temp_or_sent = []
                # marks the first term
                temp_first_mark = []
                # marks the second term
                temp_second_mark = []
                if 'first_sentence' in element and 'second_sentence' in element:
                    temp_id = len(self.final_data)
                    for word in element['first_sentence']:
                        if word not in self.word2idx:
                            curr_id = len(list(self.word2idx))
                            self.word2idx[word] = curr_id
                            self.idx2word[word] = curr_id
                    temp_sent.append(self.word2idx[word])
                    temp_or_sent.append(word)
                    for word in element['second_sentence']:
                        if word not in self.word2idx:
                            curr_id = len(list(self.word2idx))
                            self.word2idx[word] = curr_id
                            self.idx2word[word] = curr_id
                    temp_sent.append(self.word2idx[word])
                    temp_or_sent.append(word)
                    for type in element['rel']:
                        if type not in self.label2idx:
                            curr_id = len(list(self.label2idx))
                            self.label2idx[element['rel']] = curr_id
                            self.idx2word[element['rel']] = curr_id
                    dict_data['token'] = temp_sent
                    dict_data['ortoken'] = temp_or_sent
                    dict_data['rel'] = self.label2idx[element['rel']]
                    dict_data['orrel'] = element['rel']
                    dict_data['find_one'] = sublistfind(first_phrase, element['first_sentence'])
                    dict_data['find_two'] = sublistfind(second_phrase, element['second_sentence'])
                else:
                    pass
            self.final_data.append(dict_data)

    def data_split(self):
        x = []
        y = []
        z = []
        for element in self.final_data:
            if 'token' in element:
                x.append([element['ortoken']])
                #x.append([element['ortoken'], element['find_one'], element['find_two']])
                y.append([element['orrel']])
                z.append([element['ortoken'], element['orrel'], element['find_one'], element['find_two']])
        self.x = x
        self.y = y
        self.z = z

    def testEverything(self):
        # Testing by printing everything out.
        print(self.data)
        print(self.label)
        print(self.int2concept)
        print(self.concept2int)
        print(self.conceptList)
        print(self.weights)
        print(self.byConcept)

class Prereqdataset(torchtext.data.Dataset):
    def __init__(self, text_field, label_field, mooc_reader, **kwargs):
        fields = [("text",text_field), ("label", label_field)]
        examples = []
        for (i, x) in enumerate(mooc_reader.z):
            examples.append(data.Example.fromlist([x[0], x[1]], fields))
        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex) : return len(ex.text)

    @classmethod
    def splits(cls, text_field, label_field, root='.data', train='train', test='test', **kwargs):
        return super().splits(root, text_field=text-field, label_field=label_field, train=train, validation=None, test=test, **kwargs)

'''
This is the function used for reading data from the LectureBank dataset.
class LectureBankDatasetReader:
    def __init__(self):
'''

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

# helper function
def tokenize_list_words(list):
    tokenized = []
    for sent in list:
        tokenized.append(simple_preprocess(sent))
    # derived from https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    tokenized = [item for sublist in tokenized for item in sublist]
    return tokenized

# helper function
def sublistfind(pattern, _list):
    ret_list = []
    for x in _list:
        if (x in pattern):
            ret_list.append(1)
        else:
            ret_list.append(0)
    return ret_list

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = categorical_accuracy(predictions, batch.label)
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


if __name__ == "__main__":

    print('Reading from the MOOC dataset:\n')
    reader = MOOCReader()
    print('Finished reading from the MOOC dataset \n')

    # Preparation
    SEED = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    BATCH_SIZE = 32

    TEXT = data.Field(sequential = True, fix_length = 32)
    LABEL = data.LabelField(sequential = True, dtype = torch.long)
    preq = Prereqdataset(TEXT, LABEL, reader)

    print(f'The dataset has {len(preq)} trainable elements')
    train_data, test_data = preq.split()
    train_data, valid_data = train_data.split(random_state = random.seed(SEED))

    print(f'Test data has {len(test_data)} samples')
    print(f'Valid data has {len(valid_data)} samples')
    print(f'Train data has {len(train_data)} samples')
    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(train_data,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.6B.100d",
                 unk_init = torch.Tensor.normal_)

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

    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

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
