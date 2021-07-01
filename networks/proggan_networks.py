import torch, torchvision, os
from utils import proggan, customnet, util
from torch  import nn

def proggan_setting(domain):
    # default: 256 resolution, 512 z dimension, resnet 18 encoder
    outdim = 256
    nz = 512
    resnet_depth = 18
    if domain == 'celebahq-small':
        outdim = 128
    if domain == 'celebahq':
        outdim = 1024
    return dict(outdim=outdim, nz=nz, resnet_depth=resnet_depth)

def load_proggan(domain):
    # Automatically download and cache progressive GAN model
    # (From Karras, converted from Tensorflow to Pytorch.)
    if domain in ['celebahq-small', 'livingroom-paper']:
        # these are pgans we trained ourselves
        weights_filename = 'pretrained_models/pgans_%s_generator.pth' % domain
        url = 'http://latent-composition.csail.mit.edu/' + weights_filename
        sd = torch.hub.load_state_dict_from_url(url)
    else:
        # models from gan dissect
        weights_filename = dict(
            bedroom='proggan_bedroom-d8a89ff1.pth',
            church='proggan_churchoutdoor-7e701dd5.pth',
            conferenceroom='proggan_conferenceroom-21e85882.pth',
            diningroom='proggan_diningroom-3aa0ab80.pth',
            kitchen='proggan_kitchen-67f1e16c.pth',
            livingroom='proggan_livingroom-5ef336dd.pth',
            restaurant='proggan_restaurant-b8578299.pth',
            celebahq='proggan_celebhq-620d161c.pth')[domain]
        # Posted here.
        url = 'http://gandissect.csail.mit.edu/models/' + weights_filename
        try:
            sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
        except:
            sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    model = proggan.from_state_dict(sd)
    model = model.eval()
    return model


def load_proggan_encoder(domain, nz=512, outdim=256, use_RGBM=True, use_VAE=False,
                         resnet_depth=18, ckpt_path='pretrained'):
    assert not(use_RGBM and use_VAE),'specify one of use_RGBM, use_VAE'
    if use_VAE:
        nz = nz*2
    channels_in = 4 if use_RGBM or use_VAE else 3
    print(f"Using halfsize?: {outdim<150}")
    print(f"Input channels: {channels_in}")
    netE = customnet.CustomResNet(size=resnet_depth, num_classes=nz,
                                  halfsize=outdim<150,
                                  modify_sequence=customnet.modify_layers,
                                  channels_in=channels_in)
    if ckpt_path is None: # does not load weights
        return netE

    if ckpt_path == 'pretrained':
        # use the pretrained checkpoint path (RGBM model)
        assert(use_RGBM)
        assert(not use_VAE)
        suffix = 'RGBM'
        ckpt_path = f'pretrained_models/pgan_encoders_{domain}_{suffix}_model_final.pth'
        print(f"Using default checkpoint path: {ckpt_path}")
        url = 'http://latent-composition.csail.mit.edu/' + ckpt_path
        ckpt = torch.hub.load_state_dict_from_url(url)
    else:
        ckpt = torch.load(ckpt_path)
    netE.load_state_dict(ckpt['state_dict'])
    netE = netE.eval()
    return netE

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size,dilate=1, padd=0, stride=1):
        super(ConvBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd,dilation=dilate)
        self.norm1 = nn.BatchNorm2d(out_channel) #out_channel
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0,dilation=1)
        self.norm2 = nn.BatchNorm2d(out_channel) #out_channel
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0,dilation=1)
        self.norm3 = nn.BatchNorm2d(out_channel) #out_channel

    def forward(self,x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        # x = self.conv3(x)
        # x = self.norm3(x)
        return x

class MaskednetE(nn.Module):
    def __init__(self, ratio,mask_width,netE):
        super(MaskednetE, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = 32
        self.head = ConvBlock(3,N,ratio,dilate = mask_width)
        # self.head.conv1.weight[:,:,2:-2,2:-2] = 0
        self.body = nn.Sequential()
        num_layers = int(mask_width / 2)
        for i in range(num_layers):
            if i < num_layers - 1:
                ker = 3
            else:
                ker = 2
            if N < 1024:
                block = ConvBlock(N, 2 * N, ker, 1, 0, 1)
            else:
                block = ConvBlock(N, N, ker, 1, 0, 1)
            self.body.add_module('block%d'%(i+1),block)
            if N < 1024:
                N *= 2
            else:
                N = 1024
        self.tail = nn.Conv2d(N,512,kernel_size=1,stride=1,padding=0)
        self.adaptaveragepool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512,120,bias=True)
        self.to_z = netE.to_z

    def forward(self,x,mask,GAN):
        x = x*mask
        # self.head.conv1.weight[:,:,2:-2,2:-2] = 0
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        # if x.shape[2] != 1 or x.shape[3] != 1:
        #     print('check the network!!')
        x = self.adaptaveragepool(x)
        if 'BigGAN' in GAN:
            x = x.reshape(x.shape[0],x.shape[1])
            x = self.linear(x)
            x = self.to_z(x)
        return x

def adjust_netE(netE, mask_width, resolution,ratio):
    # ratio = int(resolution/mask_width)
    netE.conv1 = nn.Conv2d(3,64, kernel_size = (ratio,ratio), dilation=(ratio,ratio))
    netE.conv1.weight[:, :, 2:-2, 2:-2] = 0
    netE.conv1.weight.requires_grad = False

