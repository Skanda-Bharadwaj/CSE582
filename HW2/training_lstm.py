import time
import numpy as np
import torch
import torch.nn as nn
from data_utils import *
from data_preprocessing import map_sentiment
import torch.nn.functional as F
from nltk.corpus import stopwords
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from lstm_model import SentimentRNN

OUTPUT_FOLDER = './all_outputs/'
# nltk.download('stopwords')

input_file_path = './dataset/output/output_reviews_top.csv'
top_data_df_small = pd.read_csv(input_file_path)

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# top_data_df_small = get_dataframe(input_file_path)
# w2vmodel, word2vec_file = make_word2vec_model(top_data_df_small, OUTPUT_FOLDER, padding=True, sg=sg, min_count=min_count, size=size, workers=workers, window=window)

# function to predict accuracy
def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s


# x_train: list of all reviews

def tockenize(reviews_train_list, review_test_list):
    word_list = []

    stop_words = set(stopwords.words('english'))
    for sent in reviews_train_list:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w: i + 1 for i, w in enumerate(corpus_)}

    # tockenize
    final_list_train, final_list_test = [], []
    for sent in reviews_train_list:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in review_test_list:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])

    return final_list_train, final_list_test, onehot_dict



# train for some number of epochs
epoch_tr_loss, epoch_vl_loss = [], []
epoch_tr_acc, epoch_vl_acc = [], []

reviews = top_data_df_small['text'].values
ratings = top_data_df_small['stars'].values
ratings_sentiments = [ map_sentiment(x) for x in ratings]

X,y = reviews[:30000], ratings_sentiments[:30000]
x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)

x_train_bowVec = []
y_train_target = []

x_train,x_test,vocab = tockenize(x_train,x_test)
x_train_pad = padding_(x_train,300)
x_test_pad = padding_(x_test,300)

y_train = np.array(y_train)
print(y_train)

y_test = np.array(y_test)
print(y_test)


# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

# dataloaders
batch_size = 50

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

dataiter = iter(train_loader)
sample_x, sample_y = next(dataiter)

clip = 5
epochs = 5
valid_loss_min = np.Inf

no_layers = 2
vocab_size = len(vocab) + 1
embedding_dim = 500
output_dim = 3
hidden_dim = 256
loss_function = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)

model = SentimentRNN(no_layers,vocab_size,output_dim,hidden_dim,embedding_dim,drop_prob=0.5)

#moving to gpu
model.to(device)
print(model)

# loss and optimization functions
lr=0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model_path = OUTPUT_FOLDER + 'models/sentiment_model_30ep.pt'

start_time = time.time()
for epoch in range(epochs):
    train_losses = []
    train_acc = 0.0
    model.train()
    # initialize hidden state
    h = model.init_hidden(batch_size, device)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        model.zero_grad()
        output, h = model(inputs, h)

        # calculate the loss and perform backprop
        one_hot_labels = F.one_hot(labels)
        loss = loss_function(output.float(), one_hot_labels.float())

        loss.backward()
        train_losses.append(loss.item())
        # calculating accuracy
        _, predicted = torch.max(output.data, 1)
        accuracy = acc(predicted.float(), labels.float())
        train_acc += accuracy
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    val_h = model.init_hidden(batch_size, device)
    val_losses = []
    val_acc = 0.0
    model.eval()
    for inputs, labels in valid_loader:
        val_h = tuple([each.data for each in val_h])

        inputs, labels = inputs.to(device), labels.to(device)

        output, val_h = model(inputs, val_h)
        one_hot_labels = F.one_hot(labels)
        val_loss = loss_function(output.float(), one_hot_labels.float())

        val_losses.append(val_loss.item())

        _, predicted = torch.max(output.data, 1)
        accuracy = acc(predicted.float(), labels.float())
        val_acc += accuracy

    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = train_acc / len(train_loader.dataset)
    epoch_val_acc = val_acc / len(valid_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch + 1}')
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc * 100} val_accuracy : {epoch_val_acc * 100}')
    if epoch_val_loss <= valid_loss_min:
        torch.save(model.state_dict(), model_path)
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, epoch_val_loss))
        valid_loss_min = epoch_val_loss
    print(25 * '==')

total_time = time.time() - start_time
print(f"total time = {total_time}")

# torch.save(model.state_dict(), model_path)
print("Total parameters:", sum(p.numel() for p in model.parameters()))
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

def predict_text(text):
    word_seq = np.array([vocab[preprocess_string(word)] for word in text.split()
                         if preprocess_string(word) in vocab.keys()])
    word_seq = np.expand_dims(word_seq, axis=0)
    pad = torch.from_numpy(padding_(word_seq, 500))
    inputs = pad.to(device)
    batch_size = 1
    h = model.init_hidden(batch_size, device)
    h = tuple([each.data for each in h])
    output, h = model(inputs, h)
    return (output.item())

index = 30
print(top_data_df_small['text'][index])
print('='*70)
print(f'Actual sentiment is  : {top_data_df_small["stars"][index]}')
print('='*70)
pro = predict_text(top_data_df_small['text'][index])
status = "positive" if pro > 0.5 else "negative"
pro = (1 - pro) if status == "negative" else pro
print(f'Predicted sentiment is {status} with a probability of {pro}')

index = 32
print(top_data_df_small['text'][index])
print('='*70)
print(f'Actual sentiment is  : {top_data_df_small["stars"][index]}')
print('='*70)
pro = predict_text(top_data_df_small['text'][index])
status = "positive" if pro > 0.5 else "negative"
pro = (1 - pro) if status == "negative" else pro
print(f'predicted sentiment is {status} with a probability of {pro}')