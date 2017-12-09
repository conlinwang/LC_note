from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)




class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400) # 784 = 28x28 --> 400
        self.fc21 = nn.Linear(400, 10) # 400 --> 10  mean, mu
        self.fc22 = nn.Linear(400, 10) # 400 --> 10  var, logvar
        self.fc3 = nn.Linear(10, 400)  #  10 --> 400
        self.fc4 = nn.Linear(400, 784) # 400 --> 784

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          # print ('std.size()=', std.size())
          eps = Variable(std.data.new(std.size()).normal_())
          # print (eps.mul(std).add_(mu))
          return eps.mul(std).add_(mu) # (128L, 20L)
        else:
            # print (mu)
            return mu # (128L, 20L)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


model = VAE()
if args.cuda:
    model.cuda()


def loss_function(recon_x, x, mu, logvar, beta):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batch_size * 784
    

    return BCE + KLD * beta


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch,beta):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, float(beta))
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)) )
            
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))



def test(epoch,beta):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar, z = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar, float(beta)).data[0]
        if i == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + 'beta='+str(beta)+'.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

for beta in range(0,20):
    for epoch in range(1, args.epochs + 1):
        train(epoch,beta)
        test(epoch,beta)
        sample = Variable(torch.randn(64, 10))
        # print ('sample.size()', sample.size())
        if args.cuda:
           sample = sample.cuda()
        sample = model.decode(sample).cpu()
        save_image(sample.data.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) +'beta='+str(beta)+ '.png')

    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        if i == 0:
            recon_batch, mu, logvar, z = model(data) # got its recon_batch, mu, logvar, z

    sample = recon_batch[0] # (784L,) print (recon_batch[0].size()), Pick the 1st picture
    save_image(sample.data.view(1, 1, 28, 28),'results/the_original_pic_beta='+str(beta)+'.png')
    sample = z[0]
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(1, 1, 28, 28),'results/picture_from_z_decoded_beta='+str(beta)+'.png')

    Z_replace = np.linspace(-2.0, 2.0, num=10)
    Z_aug = []
    for i in range(0, 10):
        sample = z[0].clone()
        Z_code = sample.data.numpy()
        Z_temp = Z_code
        for j in range(0,10):
            Z_temp[i] = Z_replace[j]
            Z_aug = np.concatenate((Z_aug, Z_temp), axis=0)

    b = torch.from_numpy(Z_aug)
    sample = b.view(100,10)
    sample = sample.float()

    n = 10
    if args.cuda:
        sample = sample.cuda()
    sample = Variable(sample)
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(100, 1, 28, 28),'results/picture_from_touched_z_decoded_beta='+str(beta)+'.png', nrow=n)


        

# conlin # --------------------------------------------------------------------------

# # take out a test data from test_loader
# for i, (data, _) in enumerate(test_loader):
#     if args.cuda:
#         data = data.cuda()
#     data = Variable(data, volatile=True)
#     if i == 0:
#         recon_batch, mu, logvar, z = model(data) # got its recon_batch, mu, logvar, z

# # reconstruct it's pic from the recon_batch:
# sample = recon_batch[0] # (784L,) print (recon_batch[0].size()), Pick the 1st picture
# save_image(sample.data.view(1, 1, 28, 28),'results/the_original_pic_beta='+str(beta)+'.png')

# # reconstruct it's pic from z:
# sample = z[0]
# if args.cuda:
#     sample = sample.cuda()
# sample = model.decode(sample).cpu()
# save_image(sample.data.view(1, 1, 28, 28),'results/picture_from_z_decoded_beta='+str(beta)+'.png')


# # mainuplate z to reconstruct it's pic---------------------------------------------------
# Z_replace = np.linspace(-2.0, 2.0, num=10)
# Z_aug = []
# for i in range(0, 10):
#     sample = z[0].clone()
#     Z_code = sample.data.numpy()
#     Z_temp = Z_code
#     for j in range(0,10):
#         Z_temp[i] = Z_replace[j]
#         Z_aug = np.concatenate((Z_aug, Z_temp), axis=0)

# b = torch.from_numpy(Z_aug)
# sample = b.view(100,10)
# sample = sample.float()

# n = 10
# if args.cuda:
#     sample = sample.cuda()
# sample = Variable(sample)
# sample = model.decode(sample).cpu()
# save_image(sample.data.view(100, 1, 28, 28),'results/picture_from_touched_z_decoded_beta='+str(beta)+'.png', nrow=n)
