import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio


class MLP(nn.Module):
    def __init__(self, num_input_neurons):
        super(MLP, self).__init__()
        self.num_input_neurons = num_input_neurons
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_input_neurons, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y


def train(epoch, dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    # print(f"train_size: {size}")
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        # print(pred.shape)
        # print(y.shape)
        # print(pred)
        # print(y)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoc: {epoch}, loss: {loss.item():>7f}")


def make_annotation_train(place, train_test, hammer='small'):
    df = pd.read_csv("../annotations_home_linux.csv")
    df2 = df[(df['place'] == place) & (df['train_test'] == train_test) & (df['hammer_type'] == hammer)]
    df3 = df2.reset_index()
    df4 = df3[['path', 'file_name', 'label']]
    df4.to_csv('../train_home.csv', encoding='utf-8')


class ImpactEchoDataset(Dataset):

    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file)
        # self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    # len(ied)

    def __getitem__(self, index):
        # print("inside getitem")
        audio_sample_path = self._get_audio_sample_path(index)
        label = self.annotations.iloc[index, -1]
        label_dict = {'normal': torch.tensor([[0.]]), 'defect': torch.tensor([[1.]])}
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label_dict[label]

    def _get_audio_sample_path(self, index):
        dir_name = self.annotations.iloc[index, 1]
        # print(dir_name)
        file_name = self.annotations.iloc[index, 2]
        path = os.path.join(dir_name, file_name)
        # print(path)
        return path


if __name__ == "__main__":
    make_annotation_train('crack_1', 'train')
    training_data = ImpactEchoDataset('../train_home.csv')
    print(f"There are {len(training_data)} samples in the dataset.")
    signal, label = training_data[0]
    input_neurons = signal.shape[1]
    print(input_neurons)
    print(label.dtype)
    print(label.item())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    model = MLP(input_neurons).to(device)

    train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)

    loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 1000
    for t in range(epochs):
        train(t, train_dataloader, model, loss, optimizer)



