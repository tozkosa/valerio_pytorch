from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import librosa
import os


def load_audio_files(place, train_test, hammer='small'):
    df = pd.read_csv("annotations_linux.csv")
    df2 = df[df['place'] == place]
    df3 = df2[df2['train_test'] == train_test]

    print(len(df3))
    df3.to_csv('train.csv')

    X = []
    for i, (path, file_name, h_t) in enumerate(zip(df3['path'],
                                                   df3['file_name'],
                                                   df3['hammer_type'])):
        if h_t == hammer:
            file_path = os.path.join(path, file_name)
            signal, sr = librosa.load(file_path)
            print(f"{i}: {file_name}, sr={sr}, n_frames={len(signal)}")
            X.append(signal)
            print(file_name)

    return X, sr


class ImpactEchoDataset(Dataset):

    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    # len(ied)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    # a_list[i] -> a_list.__getitem__(i)


if __name__ == "__main__":
    X, r = load_audio_files("crack_1", "train")
