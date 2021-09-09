import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_INPUT = 784


class VAE(nn.Module):
    def __init__(self, device='cpu'):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = Encoder(device='device')
        self.decoder = Decoder(device='device')

    def forward(self, x):
        mean, var = self.encoder(x)
        z = self.reparameterize(mean, var)
        y = self.decoder(z)
        return y, z

    def reparameterize(self, mean, var):
        eps = torch.randn(mean.size()).to(self.device)
        z = mean + torch.sqrt(var) * eps
        return z

    def lower_bound(self, x):
        mean, var = self.encoder(x)
        z = self.reparameterize(mean, var)
        y = self.decoder(z)

        reconst = - torch.mean(torch.sum(x*torch.log(y)
                                         + (1-x) * torch.log(1-y), dim=1))
        kl = - 1/2 * torch.mean(torch.sum(1
                                          + torch.log(var)
                                          - mean**2
                                          - var, dim=1))
        L = reconst + kl
        return L


class Encoder(nn.Module):
    def __init__(self, device='cpu'):
        super(Encoder, self).__init__()
        self.device = device
        self.l1 = nn.Linear(NUM_INPUT, 200)
        self.l_mean = nn.Linear(200, 10)
        self.l_var = nn.Linear(200, 10)

    def forward(self, x):
        h = self.l1(x)
        h = torch.relu(h)
        mean = self.l_mean(h)
        var = self.l_var(h)
        var = F.softplus(var)
        return mean, var


class Decoder(nn.Module):
    def __init__(self,  device='cpu'):
        super(Decoder, self).__init__()
        self.device = device
        self.l1 = nn.Linear(10, 200)
        self.out = nn.Linear(200, NUM_INPUT)

    def forward(self, x):
        h = self.l1(x)
        h = torch.relu(h)
        h = self.out(h)
        y = torch.sigmoid(h)
        return y


if __name__ == '__main__':
    pass