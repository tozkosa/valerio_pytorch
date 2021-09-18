import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from daon_dataset import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=481):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        # N, 784
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
            nn.Tanh()    # changed from sigmoid in mnist
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    classes = ['normal', 'defect']
    file = '../ae.csv'

    make_annotation_train_test('crack_1', 'test', 'small', file)

    data = ImpactEchoDataset(file, classes=classes)
    print(f"There are {len(data)} samples in the dataset.")
    data_loader = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=64,
                                              shuffle=True)

    # This is to check image data.
    data_iter = iter(data_loader)
    echos, _ = data_iter.next()
    print(torch.min(echos), torch.max(echos))

    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # print(model)

    num_epochs = 20
    outputs = []
    for epoch in range(num_epochs):
        for (echo, label) in data_loader:
            echo = echo.to(device)
            recon = model(echo)
            loss = criterion(recon, echo)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch:{epoch+1}, Loss:{loss.item():.4f}")
        outputs.append((epoch, echo, recon, label))

    # save the reconstructed sounds
    recon_wave = outputs[-1][2].detach().cpu()
    recon_label = outputs[-1][3].detach().cpu().numpy()
    # data_labels = [classes[i] for i in recon_label]
    # print(recon_label)
    # print(data_labels)
    # print(recon_wave.shape)

    model.eval()
    for i, (echo, label) in enumerate(data_loader):
        echo = echo.to(device)
        latent_variables = model.encoder(echo)
        class_labels = label

    x = class_labels.numpy()
    print(f"label's shape: {x.shape}")

    colors = list(mcolors.BASE_COLORS.keys())
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # print(f"latent_variables shape: {latent_variables.shape}")
    # for i, la in enumerate(latent_variables):
    #     la = la.detach().cpu().numpy()
    #     print(la)
    #     # print(class_labels[i])
    #     ax.scatter(la[0], la[1], la[2], c=colors[x[i]])

    plt.figure()
    print(f"latent_variables shape: {latent_variables.shape}")
    for i, la in enumerate(latent_variables):
        la = la.detach().cpu().numpy()
        # print(la)
        # print(class_labels[i])
        # plt.scatter(la[0], la[1], c=colors[x[i]])
        plt.scatter(la[0], la[1])

    # for i, latent_d in enumerate(latent_variables):
    #     latent_d = latent_d.detach().cpu().numpy()
    #     label_d = class_labels[i]
    #     print(label_d.shape)
    #     # ax.scatter(latent_d[0], latent_d[1],
    #     #            latent_d[2], c=colors[label_d])

    # for i, item in enumerate(recon_wave):
    #     path = f"save_wav_test_{i+1}.wav"
    #     audio_data = item[np.newaxis, :]
    #     # print(item.shape)
    #     # print(audio_data.shape)
    #     torchaudio.save(path,
    #                     audio_data,
    #                     96000,
    #                     encoding="PCM_S", bits_per_sample=24)
    #
    path = "save_wav_test.wav"
    torchaudio.save(path,
                    outputs[-1][2].detach().cpu(),
                    96000,
                    encoding="PCM_S", bits_per_sample=24)


    for k in range(0, num_epochs, 8):
        plt.figure(figsize=(9, 2))
        plt.gray()
        echos = outputs[k][1].detach().cpu().numpy()
        recon = outputs[k][2].detach().cpu().numpy()
        for i, item in enumerate(echos):
            if i >= 9:
                break
            plt.subplot(2, 9, i+1)
            item = item.reshape(13, 37)
            plt.imshow(item)
            plt.axis('off')

        for i, item in enumerate(recon):
            if i >= 9:
                break
            plt.subplot(2, 9, 9+i+1)
            item = item.reshape(13, 37)
            plt.imshow(item)
            plt.axis('off')

    plt.show()
