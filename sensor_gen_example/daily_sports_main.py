import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils import random_split
from daily_sports_model import CNNClassifier
from daily_sports_dataset import daily_sports_dataset

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def classifier_train(train_loader, model, criterion, optimizer, num_epochs, device):
    model.train()
    print("Training the Model: ")
    for epoch in range(num_epochs):
        running_loss = 0
        running_bsz = 0
        correct = 0
        for idx, (samples, labels) in enumerate(train_loader):
            samples = samples.to(device, dtype=torch.float) # original is Input type (torch.cuda.DoubleTensor)
            labels = labels.to(device)
            bsz = labels.shape[0]

            optimizer.zero_grad()

            outputs = model(samples)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_bsz += bsz
          
            pred = outputs.argmax(dim=1)
            correct += pred.eq(labels.view_as(pred)).sum().item()

        print(f'Epoch {epoch+1} Loss: {running_loss/running_bsz:.6f}, Train Accuracy: {100. * correct / (running_bsz):.4f}')
    return model

def classifier_validate(val_loader, model, criterion):
    model.eval()
    correct = 0
    batch_cnt = 0
    with torch.no_grad():
        for idx, (samples, labels) in enumerate(val_loader):
          samples = samples.to(device, dtype=torch.float)
          labels = labels.to(device)
          bsz = labels.shape[0]
          batch_cnt += bsz

          output = model(samples)
          pred = output.argmax(dim=1)
          correct += pred.eq(labels.view_as(pred)).sum().item()
        print('Test Accuracy: {:.4f}'.format(100. * correct / (batch_cnt)))

def main(train_dataset, test_dataset, model, device, train_mode, num_epochs=60, batch_size=16, lr=3e-5):

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    if train_mode == True:
        model = classifier_train(train_loader, model, criterion, optimizer, num_epochs, device)
        print("Saving model: ")
        torch.save(model, "daily_sports_classifier.pt")
    else:
        model = torch.load("daily_sports_classifier.pt")

    classifier_validate(test_loader, model, criterion)

if __name__ == '__main__':
    
    seed = 42
    dataset_saved = True # true: load data from saved files
    train_mode = False # true: training the model from scratch

    train_ratio=0.8

    dataset = daily_sports_dataset(saved = dataset_saved)
    train_dataset, test_dataset = random_split(dataset, [train_ratio, 1-train_ratio],
    generator=torch.Generator().manual_seed(seed)) # test accuracy is 98.88

    # train_dataset = daily_sports_dataset(saved = dataset_saved, excluded_person_list=['p7', 'p8'])
    # test_dataset = daily_sports_dataset(excluded_person_list=['p1', 'p2', 'p3', 'p4', 'p5', 'p6']) # test accuracy is 72%: cross-domain problem

    model = CNNClassifier()

    main(train_dataset, test_dataset, model, device, train_mode)
