import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class Autoencoder_CNN(nn.Module):
    def __init__(self):
        super(Autoencoder_CNN, self).__init__()
        # N, 784
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # N 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),  # N, 64, 1, 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    transform = transforms.ToTensor()
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5), (0.5))
    # ])
    mnist_data = datasets.MNIST(root='../data',
                                train=True,
                                download=True,
                                transform=transform
                                )

    data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                              batch_size=4,
                                              shuffle=True)

    # This is to check image data.
    # dataiter = iter(data_loader)
    # images, labels = dataiter.next()
    # print(torch.min(images), torch.max(images))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = Autoencoder_CNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    print(model)

    num_epochs = 10
    outputs = []
    for epoch in range(num_epochs):
        for (img, _) in data_loader:
            # img = img.reshape(-1, 28*28).to(device)
            img = img.to(device)
            recon = model(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch:{epoch+1}, Loss:{loss.item():.4f}")
        outputs.append((epoch, img, recon))

    for k in range(0, num_epochs, 4):
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = outputs[k][1].detach().cpu().numpy()
        recon = outputs[k][2].detach().cpu().numpy()
        for i, item in enumerate(imgs):
            if i >= 9:
                break
            plt.subplot(2, 9, i+1)
            # item = item.reshape(-1, 28, 28)
            plt.imshow(item[0])

        for i, item in enumerate(recon):
            if i >= 9:
                break
            plt.subplot(2, 9, 9+i+1)
            # item = item.reshape(-1, 28, 28)
            plt.imshow(item[0])
    plt.show()
