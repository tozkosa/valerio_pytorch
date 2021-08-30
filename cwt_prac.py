import matplotlib.pyplot as plt
import numpy as np
import pywt
import pandas as pd
import librosa
import os


def load_audio_files(place, train_test):
    df = pd.read_csv("annotations.csv")
    df2 = df[df['place'] == place]
    df3 = df2[df2['train_test'] == train_test]

    print(len(df3))
    df3.to_csv('temp.csv')

    X = []
    for i, (path, file_name) in enumerate(zip(df3['path'], df3['file_name'])):
        file_path = os.path.join(path, file_name)
        signal, sr = librosa.load(file_path)
        print(f"{i}: {file_name}, sr={sr}, n_frames={len(signal)}")
        X.append(signal)

    return X, sr


if __name__ == "__main__":
    X, sr = load_audio_files('crack_1', 'test')
    wavlist = pywt.wavelist(kind='continuous')
    print(wavlist)
    wavelet_type = 'cmor1.5-1.0'  # 'cmor1.5-1.0'
    # wav = pywt.ContinuousWavelet(wavelet_type)
    # int_psi, x = pywt.integrate_wavelet(wav, precision=8)
    # plt.plot(x, int_psi)
    # plt.show()
    # print(len(int_psi))
    fs = sr
    nq_f = fs / 2.0
    print(fs)
    print(nq_f)

    freqs = np.linspace(1, nq_f, 50)
    freqs_rate = freqs / fs
    scales = 1 / freqs_rate
    scales = scales[::-1]
    print(len(scales))

    frequencies_rate = pywt.scale2frequency(scale=scales, wavelet=wavelet_type)
    print(frequencies_rate)

    frequencies = frequencies_rate * fs
    print(frequencies)

    for i in range(0, 64, 4):
        signal = X[i]
        cwtmatr, freqs_rate = pywt.cwt(signal, scales=scales, wavelet=wavelet_type)
        print(cwtmatr.shape)
        plt.figure()
        plt.imshow(np.abs(cwtmatr), aspect='auto')

    plt.show()
    # scale = np.arange(1, 1025) # 257
    # signal = X[2]
    # coef, _ = pywt.cwt(signal, scales=scale, wavelet=wavelet_type)
    # t = np.arange(len(signal))/sr
    # frq = pywt.scale2frequency(scale=scale, wavelet=wavelet_type)*sr
    # plt.pcolormesh(t, frq, 10*np.log(np.abs(coef)), cmap='jet')
    # plt.colorbar()
    # plt.show()
    # plt.close()




