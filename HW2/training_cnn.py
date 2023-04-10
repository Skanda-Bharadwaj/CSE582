import time
import matplotlib
from cnn_model import *
from data_utils import *
import torch.optim as optim
from sklearn.metrics import classification_report
from data_preprocessing import get_dataframe, split_train_test

def acc(pred,label):
    pred = pred.float()
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

NUM_CLASSES = 3
size = 500
window = 3
min_count = 1
workers = 3
sg = 1
OUTPUT_FOLDER = './all_outputs/'

input_file_path = './dataset/output/output_reviews_top.csv'
top_data_df_small = get_dataframe(input_file_path)
w2vmodel, word2vec_file = make_word2vec_model(top_data_df_small, OUTPUT_FOLDER, padding=True, sg=sg, min_count=min_count, size=size, workers=workers, window=window)
VOCAB_SIZE = len(w2vmodel.wv.vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)

cnn_model = CnnTextClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
print(cnn_model)

cnn_model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
num_epochs = 5

# Open the file for writing loss
X_train, X_test, Y_train, Y_test = split_train_test(top_data_df_small)
loss_file_name = OUTPUT_FOLDER + 'plots/' + 'cnn_class_big_loss_with_padding_relu.csv'
f = open(loss_file_name, 'w')
f.write('iter, loss')
f.write('\n')
losses = []
cnn_model.train()

start_time = time.time()
for epoch in range(num_epochs):
    print("Epoch" + str(epoch + 1))
    train_loss = 0
    train_acc = 0.0
    for index, row in X_train.iterrows():
        # Clearing the accumulated gradients
        cnn_model.zero_grad()

        # Make the bag of words vector for stemmed tokens
        bow_vec = make_word2vec_vector_cnn(row['stemmed_tokens'], top_data_df_small, w2vmodel, device)

        # Forward pass to get output
        probs = cnn_model(bow_vec)

        # Get the target label
        target = make_target(Y_train['sentiment'][index], device)

        # Calculate Loss: softmax --> cross entropy loss
        loss = loss_function(probs, target)
        train_loss += loss.item()

        # Getting gradients w.r.t. parameters
        loss.backward()

        probs = cnn_model(bow_vec)
        _, predicted = torch.max(probs.data, 1)

        accuracy = acc(predicted, target)
        train_acc += accuracy

        # Updating parameters
        optimizer.step()

    val_loss = 0.0
    val_acc = 0.0
    cnn_model.eval()
    for index, row in X_test.iterrows():
        bow_vec = make_word2vec_vector_cnn(row['stemmed_tokens'], top_data_df_small, w2vmodel, device)

        # Forward pass to get output
        probs = cnn_model(bow_vec)

        # Get the target label
        target = make_target(Y_test['sentiment'][index], device)

        # Calculate Loss: softmax --> cross entropy loss
        loss = loss_function(probs, target)
        val_loss += loss.item()

        probs = cnn_model(bow_vec)
        _, predicted = torch.max(probs.data, 1)

        accuracy = acc(predicted, target)
        val_acc += accuracy

    epoch_train_acc = train_acc / len(X_train)
    epoch_val_acc = val_acc / len(X_test)
    print(f"train_acc: {epoch_train_acc} | val_acc: {epoch_val_acc}")

    # if index == 0:
    #     continue
    print("Epoch ran :" + str(epoch + 1))
    f.write(str((epoch + 1)) + "," + str(train_loss / len(X_train)))
    f.write('\n')
    train_loss = 0

total_time = time.time() - start_time
print(f"total training time = {total_time}")
# torch.save(cnn_model, OUTPUT_FOLDER + 'cnn_big_model_500_with_padding.pth')

print("Total parameters:", sum(p.numel() for p in cnn_model.parameters()))
print("Trainable parameters:", sum(p.numel() for p in cnn_model.parameters() if p.requires_grad))

f.close()
# print("Input vector")
# print(bow_vec.cpu().numpy())
# print("Probs")
# print(probs)
# print(torch.argmax(probs, dim=1).cpu().numpy()[0])


########################################################################################################################

bow_cnn_predictions = []
original_lables_cnn_bow = []
cnn_model.eval()
loss_df = pd.read_csv(OUTPUT_FOLDER + 'plots/cnn_class_big_loss_with_padding_relu.csv')
print(loss_df.columns)
# loss_df.plot('loss')
with torch.no_grad():
    for index, row in X_test.iterrows():
        bow_vec = make_word2vec_vector_cnn(row['stemmed_tokens'], top_data_df_small, w2vmodel, device) ## Give other arguments
        probs = cnn_model(bow_vec)
        _, predicted = torch.max(probs.data, 1)
        bow_cnn_predictions.append(predicted.cpu().numpy()[0])
        original_lables_cnn_bow.append(make_target(Y_test['sentiment'][index], device).cpu().numpy()[0])
print(classification_report(original_lables_cnn_bow,bow_cnn_predictions))
loss_file_name = OUTPUT_FOLDER +  'plots/' + 'cnn_class_big_loss_with_padding_relu.csv'
loss_df = pd.read_csv(loss_file_name)
print(loss_df.columns)
plt_500_padding_30_epochs = loss_df[' loss'].plot()
fig = plt_500_padding_30_epochs.get_figure()
fig.savefig(OUTPUT_FOLDER +'plots/' + 'loss_plt_500_padding_5_epochs_relu.pdf')