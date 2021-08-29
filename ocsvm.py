from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_magnitude_spectrum(signal, sr, title, f_ratio=1):
    X = np.fft.fft(signal)
    X_mag = np.absolute(X)
    plt.figure(figsize=(18,5))
    f = np.linspace(0, sr, len(X_mag))
    f_bins = int(len(X_mag) * f_ratio)
    plt.plot(f[:f_bins], X_mag[:f_bins])
    plt.xlabel('Frequency (Hz)')
    plt.title(title)
    # plt.show()


def get_spectrum_magnitude(X, sr, f_ratio=0.5):
    f = np.linspace(0, sr, len(X[0]))
    f_bins = int(len(X[0]) * f_ratio)
    for i in range(len(X)):
        ft = np.fft.fft(X[i])
        ft_mag = np.absolute(ft)
        X[i] = ft_mag[:f_bins]
    return X


if __name__ == "__main__":
    # X = [[0], [0.44], [0.45], [0.46], [1]]
    # clf = OneClassSVM(gamma='scale').fit(X)
    # print(clf.predict(X))
    # print(clf.score_samples(X))

    df = pd.read_csv("annotations.csv")
    df2 = df[df['place'] == 'crack_2']
    df3 = df2[df2['train_test'] == 'test']

    print(len(df3))

    X = []
    for i, (path, file_name) in enumerate(zip(df3['path'], df3['file_name'])):
        file_path = os.path.join(path, file_name)
        signal, sr = librosa.load(file_path)
        print(f"{i}: {file_name}, sr={sr}, n_frames={len(signal)}")
        X.append(signal)

    """MinMaxScaler"""
    # scaler = MinMaxScaler()
    # scaler.fit(X)
    # X_scaled = scaler.transform(X)

    """Fast Fourier Transform"""
    # for i in range(5):
    #     plot_magnitude_spectrum(X[i], sr, 'FFT', 0.5)
    #     plt.show()
    # print(sr)
    # print(f"frequency bin {len(signal)}")

    X = get_spectrum_magnitude(X, sr, 0.5)
    print(len(X))
    print(len(X[0]))

    """Principal Component Analysis"""
    # pca = PCA(n_components=10)
    # X_scaled = pca.fit_transform(X)

    """One Class SVM"""
    clf = OneClassSVM(gamma='scale').fit(X_scaled)
    print(clf.predict(X_scaled))
    Y = clf.score_samples(X_scaled)
    Y2 = clf.decision_function(X_scaled)
    print(len(Y2))
    print(Y2)
    df3['score'] = list(Y2)
    #
    df3.to_csv('oneclass.csv')








