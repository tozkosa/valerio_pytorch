from torch.utils.data import Dataset
import pandas as pd
import torchaudio


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
        fold = f"fold{self.annotations.iloc{index, 5}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    # a_list[i] -> a_list.__getitem__(i)
