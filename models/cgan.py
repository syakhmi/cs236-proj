import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn
from tqdm import tqdm
from models.interface import ConditionedGenerativeModel

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class CDCGAN_D(nn.Module):
    # initializers
    def __init__(self, embed_dim=10, n_filters=128):
        super(CDCGAN_D, self).__init__()
        self.conv1_1 = nn.Conv2d(3, n_filters//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(embed_dim, n_filters//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(n_filters*2)
        self.conv3 = nn.Conv2d(n_filters*2, n_filters*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(n_filters*4)
        self.conv4 = nn.Conv2d(n_filters*4, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, embed):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(embed), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))

        return x

class CDCGAN_G(nn.Module):
    def __init__(self, z_dim=100, embed_dim=10, n_filters=128):
        super(CDCGAN_G, self).__init__()
        self.z_dim = z_dim
        self.deconv1_1 = nn.ConvTranspose2d(z_dim, n_filters*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(n_filters*2)
        self.deconv1_2 = nn.ConvTranspose2d(embed_dim, n_filters*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(n_filters*2)
        self.deconv2 = nn.ConvTranspose2d(n_filters*4, n_filters*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(n_filters*2)
        self.deconv3 = nn.ConvTranspose2d(n_filters*2, n_filters, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(n_filters)
        self.deconv4 = nn.ConvTranspose2d(n_filters, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, embed):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(embed)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        
        return x

    def sample(self, captions_embd):
        '''
        :param captions_embd: torch.FloatTensor bsize * embd_size
        :return: imgs : torch.FloatTensor of size n_imgs * c * h * w
        '''

        z = torch.randn(captions_embd.shape[0], self.z_dim, 1, 1).to(captions_embd.device)
        with torch.no_grad():
            gen_imgs = self.forward(z, captions_embd.unsqueeze(dim=2).unsqueeze(dim=3))
        gen_imgs = (gen_imgs + 1) / 2

        return gen_imgs