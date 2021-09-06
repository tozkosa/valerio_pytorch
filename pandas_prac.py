import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio


def make_annotation_train(place, train_test, hammer='small'):
    df = pd.read_csv("annotations_linux.csv")
    print(df)

    df2 = df[df['place'] == place]
    df3 = df2[df2['train_test'] == train_test]
    df4 = df3[df3['hammer_type'] == hammer].reset_index()
    df5 = df4[['path', 'file_name', 'label']]
    print(f"df5: {df5}")

    df5.to_csv('train.csv', encoding='utf-8')

    # df3 = df2[df2['train_test'] == 'test']
    # # df3 = df[['file_name', 'place', 'label']]
    # print(df3.head(5))


class ImpactEchoDataset(Dataset):

    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file)
        # self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    # len(ied)

    def __getitem__(self, index):
        print("inside getitem")
        audio_sample_path = self._get_audio_sample_path(index)
        label = self.annotations.iloc[index, -1]
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label

    def _get_audio_sample_path(self, index):
        dir_name = self.annotations.iloc[index, 1]
        print(dir_name)
        file_name = self.annotations.iloc[index, 2]
        path = os.path.join(dir_name, file_name)
        print(path)
        return path


if __name__ == "__main__":
    make_annotation_train('crack_1', 'train')
    training_data = ImpactEchoDataset('train.csv')
    print(f"There are {len(training_data)} samples in the dataset.")
    signal, label = training_data[6]
    print(signal)

    a = 1
    # train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)


