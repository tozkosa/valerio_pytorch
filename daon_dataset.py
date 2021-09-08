from torch.utils.data import Dataset
import pandas as pd
import torch
import torchaudio
import librosa
import os


def make_annotation_train_test(place, train_test, hammer, file_name):
    df = pd.read_csv("../annotations_linux.csv")
    df2 = df[(df['place'] == place) & (df['train_test'] == train_test) & (df['hammer_type'] == hammer)]
    # df3 = df2.reset_index()
    df4 = df2[['path', 'file_name', 'label']]
    df4.to_csv(file_name, encoding='utf-8')


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
        if label_ in self.classes:
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
    file_train = '../train.csv'
    file_test = '../test.csv'
    make_annotation_train_test('crack_1', 'train', 'small', file_train)
    make_annotation_train_test('crack_2', 'test', 'small', file_test)
    training_data = ImpactEchoDataset(file_train, classes=classes)
    test_data = ImpactEchoDataset(file_test, classes=classes)
    print(f"There are {len(training_data)} samples in the training dataset.")

    signal, label = training_data[40]
    input_neurons = len(signal)
    print(f"number of input neurons: {input_neurons}")
    # print(label.dtype)
    print(label)
    print(classes[label])
