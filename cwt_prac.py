import matplotlib.pyplot as plt
import numpy as np
import pywt
import pandas as pd
import librosa
import os


def load_audio_files(place, train_test, hammer='small'):
    df = pd.read_csv("annotations.csv")
    df2 = df[df['place'] == place]
    df3 = df2[df2['train_test'] == train_test]

    print(len(df3))
    df3.to_csv('temp.csv')

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


if __name__ == "__main__":
    X, sr = load_audio_files('crack_1', 'train', 'small')
    wavlist = pywt.wavelist(kind='continuous')
    # print(wavlist)
    wavelet_type = 'cmor1.5-1.0'  # 'cmor1.5-1.0'
    # wav = pywt.ContinuousWavelet(wavelet_type)
    # int_psi, x = pywt.integrate_wavelet(wav, precision=8)
    # plt.plot(x, int_psi)
    # plt.show()
    # print(len(int_psi))
    fs = sr
    nq_f = fs / 2.0
    # print(fs)
    # print(nq_f)

    freqs = np.linspace(1, nq_f, 50)
    freqs_rate = freqs / fs
    scales = 1 / freqs_rate
    scales = scales[::-1]
    # print(len(scales))

    frequencies_rate = pywt.scale2frequency(scale=scales, wavelet=wavelet_type)
    # print(frequencies_rate)

    frequencies = frequencies_rate * fs
    # print(frequencies)

    if len(X) > 20:
        num = 20
    else:
        num = len(X)

    for i in range(0, num, 4):
        signal = X[i]
        cwtmatr, freqs_rate = pywt.cwt(signal, scales=scales, wavelet=wavelet_type)
        # print(cwtmatr.shape)
        # plt.subplot(1, 5, i)
        plt.figure()
        plt.imshow(np.log10(np.abs(cwtmatr)), aspect='auto')

    plt.show()
    plt.close()





