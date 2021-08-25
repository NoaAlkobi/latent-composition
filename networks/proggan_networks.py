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
                         resnet_depth=18, ckpt_path='pretrained',which_model='orig'):
    assert not(use_RGBM and use_VAE),'specify one of use_RGBM, use_VAE'
    if use_VAE:
        nz = nz*2

    channels_in = 4 if use_RGBM or use_VAE else 3
    print(f"Using halfsize?: {outdim<150}")
    print(f"Input channels: {channels_in}")
    if 'orig' in which_model:
        netE = customnet.CustomResNet(size=resnet_depth, num_classes=nz,
                                      halfsize=outdim<150,
                                      modify_sequence=customnet.modify_layers,
                                      channels_in=channels_in)
    elif 'unet' in which_model:
        netE = UNET_encoder(nz)
    elif 'insert_class' in which_model:
        netE = customnet.CustomResNetInsertClass(size=resnet_depth, num_classes=nz,
                                      halfsize=outdim<150,
                                      modify_sequence=customnet.modify_layers,
                                      channels_in=channels_in)
    else:
        netE = customnet.CustomResNetNoPadding(size=resnet_depth, num_classes=nz,
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
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd,dilation=dilate) #add bias
        self.norm1 = nn.BatchNorm2d(out_channel) #out_channel
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0,dilation=1)#add bias
        self.norm2 = nn.BatchNorm2d(out_channel) #out_channel
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0,dilation=1)#add bias
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
        num_layers = int(mask_width / 2) - 1
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
        self.tail = nn.Conv2d(N,512,kernel_size=3,stride=1,padding=0) #add bias
        self.adaptaveragepool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512,120,bias=True) #,bias=True
        self.to_z = netE.to_z

    def forward(self,x,mask,mask_width,GAN):
        x = x * mask
        # x[:,:,mask_width:-mask_width,mask_width:-mask_width] = 0
        self.head.conv1.weight[:,:,1:-1,1:-1].data.fill_(0.0)
        if self.head.conv1.weight[:,:,1:-1,1:-1].sum() != 0:
            e=3
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        # if x.shape[2] != 1 or x.shape[3] != 1:
        #     print('check the network!!')
        # x = self.adaptaveragepool(x)
        if 'BigGAN' in GAN:
            x = x.reshape(x.shape[0],x.shape[1])
            x = self.linear(x)
            x = self.to_z(x)
        return x


class MyConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, dilate):
        super(MyConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, dilation=dilate, padding=0)
        # self.norm = nn.BatchNorm2d(out_channel) #out_channel
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self,x):
        x = self.conv(x)
        # x = self.norm(x)
        x = self.relu(x)
        # x = self.conv(x)
        return x

class Encoder_loss_network(nn.Module):
    def __init__(self, resolution):
        super(Encoder_loss_network, self).__init__()

        self.block1 = MyConvBlock(in_channel=3, out_channel=32, ker_size=5, dilate=1)
        self.block2 = MyConvBlock(in_channel=32, out_channel=32, ker_size=5, dilate=2)
        self.block3 = MyConvBlock(in_channel=32, out_channel=64, ker_size=5, dilate=4)
        self.block4 = MyConvBlock(in_channel=64, out_channel=64, ker_size=5, dilate=8)
        self.block5 = MyConvBlock(in_channel=64, out_channel=128, ker_size=5, dilate=16)
        self.block6 = MyConvBlock(in_channel=128, out_channel=128, ker_size=5, dilate=32)
        # self.block7 = MyConvBlock(in_channel=128, out_channel=128, ker_size=5, dilate=4)
        self.conv = nn.Conv2d(128, 1, kernel_size=4, padding=0)


    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        # x = self.block7(x)
        x = self.conv(x)
        return x

class UNET_encoder(nn.Module):
    def __init__(self, nz):
        super(UNET_encoder, self).__init__()

        self.block1 = UNETConvBlockDown(in_channel=3, out_channel=32, ker_size=3,stride=2)
        self.block2 = UNETConvBlockDown(in_channel=32, out_channel=64, ker_size=3,stride=2)
        self.block3 = UNETConvBlockDown(in_channel=64, out_channel=64, ker_size=3,stride=2)
        self.block4 = UNETConvBlockDown(in_channel=64, out_channel=32, ker_size=3,stride=2)

        self.up = nn.Sequential()
        self.up.add_module('block5',UNETConvBlockUp(in_channel=32, out_channel=64))
        self.up.add_module('block6',UNETConvBlockUp(in_channel=64, out_channel=128))
        if nz < 256:
            self.up.add_module('block7',UNETConvBlockUp(in_channel=128, out_channel=nz))
            self.up.add_module('block8', UNETConvBlockUp(in_channel=nz, out_channel=nz))
        else:
            self.up.add_module('block7', UNETConvBlockUp(in_channel=128, out_channel=256))
            self.up.add_module('block8',UNETConvBlockUp(in_channel=256, out_channel=nz))

        # self.block5 = UNETConvBlockUp(in_channel=32, out_channel=64)
        # self.block6 = UNETConvBlockUp(in_channel=64, out_channel=128)
        # self.block7 = UNETConvBlockUp(in_channel=128, out_channel=256)
        # self.block8 = UNETConvBlockUp(in_channel=256, out_channel=nz)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.mean(dim=(2, 3))
        x = x.reshape((x.shape[0], x.shape[1], 1, 1))
        x = self.up(x)
        # x = self.block6(x)
        # x = self.block7(x)
        # x = self.block8(x)
        return x

class UNETConvBlockDown(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size,stride, dilate=1):
        super(UNETConvBlockDown, self).__init__()
        padd = int((ker_size-1)/2)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, dilation=dilate, padding=padd)
        self.norm = nn.BatchNorm2d(out_channel) #out_channel
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.avgpool = nn.AvgPool2d(kernel_size=2,stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        # x = self.avgpool(x)

        return x

class UNETConvBlockUp(nn.Sequential):
    def __init__(self, in_channel, out_channel, dilate=1):
        super(UNETConvBlockUp, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, dilation=1, padding=0)
        self.norm = nn.BatchNorm2d(out_channel) #out_channel
        self.relu = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)


        return x