import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from matplotlib import pyplot as plt


class MLP(nn.Module):
    def __init__(self, num_input_neurons):
        super(MLP, self).__init__()
        self.num_input_neurons = num_input_neurons
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_input_neurons, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(256, 2),
            # nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # print(f"train_size: {size}")
    model.train()
    loss_batch = 0
    train_acc = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_batch += loss.item()
        train_acc += (pred.max(1)[1] == y).sum().item()
        # print(pred)
        # print(y)

        # if batch % 10 == 0:
        #     print(f"step: {batch}, loss: {loss.item():>7f}")
    loss_batch /= size
    train_acc /= size
    # loss_batch = loss.item()
    return loss_batch, train_acc


def make_annotation_train(place, train_test, hammer='small'):
    df = pd.read_csv("../annotations_home_linux.csv")
    df2 = df[(df['place'] == place) & (df['train_test'] == train_test) & (df['hammer_type'] == hammer)]
    df3 = df2.reset_index()
    df4 = df3[['path', 'file_name', 'label']]
    df4.to_csv('../train_home.csv', encoding='utf-8')


class ImpactEchoDataset(Dataset):

    def __init__(self, annotations_file, classes):
        self.annotations = pd.read_csv(annotations_file)
        self.classes = classes
        # self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    # len(ied)

    def __getitem__(self, index):
        # print("inside getitem")
        audio_sample_path = self._get_audio_sample_path(index)
        label_ = self.annotations.iloc[index, -1]
        label_ = self.classes.index(label_)
        # label_dict = {'normal': torch.tensor([[0.]]), 'defect': torch.tensor([[1.]])}
        signal_, sr = torchaudio.load(audio_sample_path)
        signal_ = signal_.view(-1)
        # print(f"signal: {signal_.shape}")
        return signal_, label_        # label_dict[label]

    def _get_audio_sample_path(self, index):
        dir_name = self.annotations.iloc[index, 1]
        # print(dir_name)
        file_name = self.annotations.iloc[index, 2]
        path = os.path.join(dir_name, file_name)
        # print(path)
        return path


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    classes = ['normal', 'defect']
    make_annotation_train('crack_1', 'train')
    training_data = ImpactEchoDataset('../train_home.csv', classes=classes)
    print(f"There are {len(training_data)} samples in the dataset.")

    signal, label = training_data[40]
    input_neurons = 481
    print(f"number of input neurons: {input_neurons}")
    # print(label.dtype)
    print(label)
    print(classes[label])

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    for X, y in train_dataloader:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}, {y.dtype}, class: {y}")
        # for i, label in enumerate(y):
        #     print(classes[label])
        break

    model = MLP(input_neurons).to(device)
    # print(model)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    train_loss_list = []
    train_acc_list = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n ----------")
        loss_batch, train_acc = train(train_dataloader, model, loss, optimizer)
        print(f"loss {loss_batch:>7f} train_acc: {train_acc:7>f}")
        train_loss_list.append(loss_batch)
        train_acc_list.append(train_acc)

    plt.figure()
    plt.plot(range(1, epochs+1), train_loss_list)
    plt.figure()
    plt.plot(range(1, epochs+1), train_acc_list)
    plt.ylim([0, 1.2])
    plt.show()






