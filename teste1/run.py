from classifier import Classifier
from readFile import fileReader
import os, pandas, numpy,torch
from torch import nn,optim
import torch.utils.data as tud

#
#Read File
#
path = os.getcwd().split('/')
path.pop()
path = '/'.join(path)+'/'
reader  =  fileReader(path+"Datasets/tweets_hate_speech.csv",path+"Datasets/NAACL_SRW_2016.csv")
data,target = reader.readData()
print(path)
#
# Baseado no código da aula de pytorch de Heike Adel dada no congresso RANLP2019
#


# loading data

# path = "/content/drive/My Drive/summerschool_data/"
# train_set = HateSpeechDataset(path + "train.data2.npy", 
#                               path + "train.length.npy", 
#                               path + "train.labels.npy")
# dev_set = HateSpeechDataset(path + "dev.data2.npy", 
#                             path + "dev.length.npy", 
#                             path + "dev.labels.npy")
# test_set = HateSpeechDataset(path + "test.data2.npy", 
#                              path + "test.length.npy", 
#                              path + "test.labels.npy")
pretrained_embeddings = torch.tensor(numpy.load(path + "Embeddings/"+"embeddings.npy"))

# setting hyperparameters
options = {
                'vocab_size': 23078,
                'emb_dim': 300,
                'num_labels': 3,
                'hidden_lstm_dim': 200,
                'bidirectional': True,
                'num_layers': 2,
                'num_epochs': 20,
                'batch_size': 100
          }

# initializing model and defining loss function and optimizer
model = Classifier(options, pretrained_embeddings)
# model.cuda()  # do this before instantiating the optimizer
loss_function = nn.NLLLoss()
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)

# training loop
train_loader = tud.DataLoader(train_set, batch_size=options['batch_size'], 
                              shuffle=True)
dev_loader = tud.DataLoader(dev_set, batch_size=options['batch_size'])
test_loader = tud.DataLoader(test_set, batch_size=options['batch_size'])

for epoch in range(options['num_epochs']):
        print("training epoch", epoch + 1)
        # train model on training data
        for data in train_loader:
                optimizer.zero_grad()
                x, l, y = sort_by_length(data['x'], data['length'], data['y'])
                scores = model(x, l)
                labels = y
                loss = loss_function(scores, labels)
                loss.backward()
                optimizer.step()

        evaluate_model(train_loader, model, "train", sort=True)
        # monitor performance on dev data
        evaluate_model(dev_loader, model, "dev", sort=True)

evaluate_model(test_loader, model, "test", sort=True)