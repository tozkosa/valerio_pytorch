import os
import json
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd


# DATA_ROOT = "D:\daon_data\\nagoya_20210727_cutout_5ms" # home
DATA_ROOT = "/home/tozeki/daon/nagoya_20210727_cutout_5ms"
JSON_PATH = "data.json"


def list_of_dirs(data_root):
    num_files = 0
    for curDir, dirs, files in os.walk(data_root):
        print("==========")
        print("Current Directory: " + curDir)
        print(f"Number of directories: {len(dirs)}")
        for dir in dirs:
            print(dir)

        cur_files = []
        for file in files:
            if file.endswith(".wav"):
                cur_files.append(file)
        print(f"Number of files: {len(cur_files)}")
        num_files += len(cur_files)

        print(cur_files)

    print(f"Total number of files: {num_files}")


def check_dirs_files(dataset_path, json_path):
    data = {
        "places": []
    }
    for curDir, dirnames, filenames in os.walk(dataset_path):
        print(curDir)
        if curDir is not dataset_path:
            place = curDir.split("\\")[-1]
            data["places"].append(place)

    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)


def select_data(dataset_path, selected, hammer):
    data_path = os.path.join(dataset_path, selected, "train", hammer)
    for curDir, dirnames, filenames in os.walk(data_path):
        file_list = []
        if curDir is not data_path:
            label = curDir.split("\\")[-1]
            for filename in filenames:
                filepath = os.path.join(curDir, filename)
                print(filepath)
                signal, sr = torchaudio.load(filepath)
                print(f"sampling rate = {sr}")
                print(f"signal shape = {signal.shape}")
                file_list.append(filepath)
    return file_list


if __name__ == "__main__":
    # list_of_dirs(DATA_ROOT)
    # check_dirs_files(DATA_ROOT, JSON_PATH)
    selected_place = "crack_1"
    type_of_hammer = "big"
    file_list = select_data(DATA_ROOT, selected_place, type_of_hammer)
    signal, sr = torchaudio.load(file_list[0])
    plt.figure()
    plt.plot(signal.t().numpy())

    specgram = torchaudio.transforms.MelSpectrogram()(signal)
    plt.figure()
    plt.imshow(specgram.log2()[0, :, :].detach().numpy(), cmap='gray')

    plt.show()
