import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from daon_dataset import *


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
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
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
        for (echo, _) in data_loader:
            echo = echo.to(device)
            recon = model(echo)
            loss = criterion(recon, echo)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch:{epoch+1}, Loss:{loss.item():.4f}")
        outputs.append((epoch, echo, recon))

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
