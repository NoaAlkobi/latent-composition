from __future__ import print_function
###trial
import os
import random
##noa
import matplotlib.axes
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from matplotlib import patches
from torch.utils.tensorboard import SummaryWriter
import oyaml as yaml
import my_resnet as resnet
# import cv2
from utils import pbar, util, masking, losses, training_utils
from networks import networks, proggan_networks
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import torchvision.models as modelsTorchVision
from torchvision import transforms
import glob

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    inp = np.clip(inp,0,1)
    return inp

def display_results(opt,resolution,fake_im,regenerated,folder_to_save,epoch,step,mask_width=0,predicted_classes=[0],hints_fake=0):
    plt.figure(figsize=(20, 10))
    if not opt.masked:
        figure, ax = plt.subplots(2,4)
        ax[0,0].imshow(convert_image_np(fake_im[0, :, :, :].reshape((1, 3, resolution, resolution))))
        ax[0,0].set_title('real image (input)')
        ax[0,0].axis('off')
        ax[1,0].imshow(convert_image_np(regenerated[0, :, :, :].detach().reshape((1, 3, resolution, resolution))))
        ax[1,0].axis('off')
        if fake_im.shape[0] > 1:
            ax[0,1].imshow(convert_image_np(fake_im[1, :, :, :].reshape((1, 3, resolution, resolution))))
            ax[0,1].axis('off')

            ax[1,1].imshow(convert_image_np(regenerated[1, :, :, :].detach().reshape((1, 3, resolution, resolution))))
            ax[1,1].axis('off')
            if fake_im.shape[0] > 2:
                ax[0, 2].imshow(convert_image_np(fake_im[2, :, :, :].reshape((1, 3, resolution, resolution))))

                ax[0,2].axis('off')
                ax[1,2].imshow(convert_image_np(regenerated[2, :, :, :].detach().reshape((1, 3, resolution, resolution))))
                ax[1,2].axis('off')

                ax[0,3].imshow(convert_image_np(fake_im[3, :, :, :].reshape((1, 3, resolution, resolution))))
                ax[0,3].axis('off')
                ax[1,3].imshow(convert_image_np(regenerated[3, :, :, :].detach().reshape((1, 3, resolution, resolution))))
                ax[1,3].axis('off')
        if len(predicted_classes) > 1:
            ax[1, 0].set_title('G(z) %d' % predicted_classes[0])
            ax[1, 1].set_title('%d' % predicted_classes[1])
            if fake_im.shape[0] > 2:
                ax[1, 2].set_title('%d' % predicted_classes[2])
                ax[1, 3].set_title('%d' % predicted_classes[3])
        else:
            ax[1,0].set_title('G(z)')
        if mask_width != 0:
            rect = patches.Rectangle((mask_width, mask_width), resolution - 2 * mask_width, resolution - 2 * mask_width, edgecolor='r', facecolor="none")
            ax[0,0].add_patch(rect)
            rect = patches.Rectangle((mask_width, mask_width), resolution - 2 * mask_width, resolution - 2 * mask_width, edgecolor='r', facecolor="none")
            ax[1,0].add_patch(rect)

            if fake_im.shape[0] > 1:
                rect = patches.Rectangle((mask_width, mask_width), resolution - 2 * mask_width,
                                         resolution - 2 * mask_width,
                                         edgecolor='r', facecolor="none")
                ax[0, 1].add_patch(rect)
                rect = patches.Rectangle((mask_width, mask_width), resolution - 2 * mask_width, resolution - 2 * mask_width,
                                         edgecolor='r', facecolor="none")
                ax[1, 1].add_patch(rect)
                if fake_im.shape[0] > 2:
                    rect = patches.Rectangle((mask_width, mask_width), resolution - 2 * mask_width, resolution - 2 * mask_width,
                                             edgecolor='r', facecolor="none")
                    ax[0, 2].add_patch(rect)
                    rect = patches.Rectangle((mask_width, mask_width), resolution - 2 * mask_width, resolution - 2 * mask_width,
                                             edgecolor='r', facecolor="none")
                    ax[0, 3].add_patch(rect)


                    rect = patches.Rectangle((mask_width, mask_width), resolution - 2 * mask_width, resolution - 2 * mask_width,
                                             edgecolor='r', facecolor="none")
                    ax[1, 2].add_patch(rect)
                    rect = patches.Rectangle((mask_width, mask_width), resolution - 2 * mask_width, resolution - 2 * mask_width,
                                             edgecolor='r', facecolor="none")
                    ax[1, 3].add_patch(rect)

    else:
        plt.subplot(3, 4, 1)
        plt.imshow(convert_image_np(fake_im[0, :, :, :].reshape((1, 3, resolution, resolution))))
        plt.title('real image (input)')
        plt.axis('off')
        plt.subplot(3, 4, 5)
        plt.imshow(convert_image_np(hints_fake[0, :, :, :].detach().reshape((1, 3, resolution, resolution))))
        plt.title('masked image')
        plt.axis('off')
        plt.subplot(3, 4, 9)
        plt.imshow(convert_image_np(regenerated[0, :, :, :].detach().reshape((1, 3, resolution, resolution))))
        plt.title('generated image G(z)')
        plt.axis('off')
        plt.subplot(3, 4, 2)
        plt.imshow(convert_image_np(fake_im[1, :, :, :].reshape((1, 3, resolution, resolution))))
        plt.axis('off')
        plt.subplot(3, 4, 6)
        plt.imshow(convert_image_np(hints_fake[1, :, :, :].detach().reshape((1, 3, resolution, resolution))))
        plt.axis('off')
        plt.subplot(3, 4, 10)
        plt.imshow(convert_image_np(regenerated[1, :, :, :].detach().reshape((1, 3, resolution, resolution))))
        plt.axis('off')
        if fake_im.shape[0]>2:
            plt.subplot(3, 4, 3)
            plt.imshow(convert_image_np(fake_im[2, :, :, :].reshape((1, 3, resolution, resolution))))
            plt.axis('off')
            plt.subplot(3, 4, 7)
            plt.imshow(convert_image_np(hints_fake[2, :, :, :].detach().reshape((1, 3, resolution, resolution))))
            plt.axis('off')
            plt.subplot(3, 4, 11)
            plt.imshow(convert_image_np(regenerated[2, :, :, :].detach().reshape((1, 3, resolution, resolution))))
            plt.axis('off')
            plt.subplot(3, 4, 4)
            plt.imshow(convert_image_np(fake_im[3, :, :, :].reshape((1, 3, resolution, resolution))))
            plt.axis('off')
            plt.subplot(3, 4, 8)
            plt.imshow(convert_image_np(hints_fake[3, :, :, :].detach().reshape((1, 3, resolution, resolution))))
            plt.axis('off')
            plt.subplot(3, 4, 12)
            plt.imshow(convert_image_np(regenerated[3, :, :, :].detach().reshape((1, 3, resolution, resolution))))
            plt.axis('off')

    plt.savefig('%s/images_epoch_%d_step_%d.jpg' % (folder_to_save, epoch, step))
    plt.close()

def vgg_output(input,model,stop_layer=29):
    slices = [3,8,15,22,29]
    x = input
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Normalize(norm_mean, norm_std)
    ])
    slices_output = []
    x = transform(x)
    for name, layer in enumerate(model.features):
        # if not isinstance(layer, torch.nn.MaxPool2d):
        x = layer(x)
        if name in slices:
            slices_output.append(x)
        if name == stop_layer: # layer_8 = RF 14*14
            break
    return slices_output



def train(opt):
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%d" % opt.gpu_num if has_cuda else "cpu")
    batch_size = int(opt.batchSize)
    resolution = opt.resolution
    cudnn.benchmark = True
    # now = datetime.now()
    if 'masked_netE' in opt.continue_learning:
        opt.masked_netE = 1
    if 'masked_losses' in opt.continue_learning:
        opt.mask_in_loss = 1
    if opt.mask_in_loss or opt.masked_netE:
        rectangle_mask = torch.ones((batch_size,1,resolution,resolution)).to(device)
        # i = opt.mask_width_percent
        mask_width = opt.mask_width
        ratio = resolution / mask_width
        if np.ceil(ratio) != np.floor(ratio):
            while np.ceil(ratio) != np.floor(ratio):
                mask_width += 1
                # mask_width = int(resolution / i)
                ratio = resolution / mask_width
        ratio = int(ratio)

        rectangle_mask[:,:,mask_width:-mask_width,mask_width:-mask_width] = 0
        if opt.mask_in_loss:
            pixel_ratio = (resolution ** 2) / rectangle_mask[0,0,:,:].sum()

    else:
        mask_width = 0
    if opt.DEBUG_PERCEPTUAL: # or opt.mask_in_loss:
        opt.vggModel = modelsTorchVision.vgg19()
        opt.vggModel.load_state_dict(torch.load("vgg19-dcbb9e9d.pth"))
        for param in opt.vggModel.parameters():
            param.requires_grad_(False)
        opt.vggModel.eval()
        opt.vggModel.to(device)
    if not opt.continue_learning == '':
        if not len(glob.glob('training/' + opt.continue_learning + '/checkpoints/netE*.pth')) > 0:
            opt.continue_learning = ''
            print('No netE saved in the folder')
    if opt.continue_learning == '':
        if opt.scenario_name=='':
            folder_to_save = 'training/%s' % (opt.GAN)
        else:
            folder_to_save = 'training/%s_%s' % (opt.scenario_name,opt.GAN)
        if opt.padding == 0:
            folder_to_save += '_no_padding'
        if opt.stride == 1:
            folder_to_save += '_stride_1'
        if opt.DEBUG_PERCEPTUAL:
            opt.lambda_lpips = 1/50
            folder_to_save += '_DEBUG_PERCEPTUAL'
        if opt.mask_in_loss:
            opt.small_RF_lpips = 1
            # opt.losses='MSE_PERCEPTUAL_norm1'
            folder_to_save += '_masked_%d_losses' % opt.mask_width
        else:
            folder_to_save += '_losses'
        if 'MSE' in opt.losses:
            folder_to_save += '_%.2f_MSE' % opt.lambda_mse
        if 'PERCEPTUAL' in opt.losses:
            folder_to_save += '_%.2f_PERCEPTUAL' % opt.lambda_lpips
            if opt.small_RF_lpips:
                folder_to_save += '_small_RF_14_14_lpips'
        if 'Z' in opt.losses:
            folder_to_save += '_%.2f_Z' % opt.lambda_latent
        if 'norm0' in opt.losses:
            folder_to_save += '_%.2f_norm_0' % opt.lambda_z_norm
        if 'norm1' in opt.losses:
            folder_to_save += '_%.2f_norm_1' % opt.lambda_z_norm
        if 'normC' in opt.losses:
            folder_to_save += '_%.2f_norm_C_' % opt.lambda_c_norm
        if opt.masked:
            folder_to_save += '_masked_model'
        if 'BigGAN' in opt.GAN:
            np.random.seed(opt.seed)
            class_number = np.random.randint(low=0, high=999, size=(opt.number_of_classes,))
            if opt.number_of_classes < 3:
                class_number[0] = 208
                if opt.number_of_classes > 1:
                    class_number[1] = 153
        folder_to_save += '_number_of_classes_%d' % opt.number_of_classes
        # for q in class_number:
        #     folder_to_save += '_%d' % q
        if 'insert_class' in opt.scenario_name:
            opt.predict_class = 0
        if opt.predict_class:
            folder_to_save += '_predict_class'
            if opt.predict_class == 1:
                folder_to_save += '_one_hot_one_num'
            elif opt.predict_class == 2:
                folder_to_save += '_one_hot_all_vec_lambda_%f' % opt.lambda_c
            elif opt.predict_class == 3:
                folder_to_save += '_c_shared_unsupervised'
            elif opt.predict_class == 4:
                folder_to_save += '_c_shared_lambda_%f' % opt.lambda_c
        if opt.masked_netE:
            folder_to_save += '_masked_netE_mask_width_%d' % mask_width
        # folder_to_save += 'training_%s' % str(now.strftime("%d_%m_%Y_%H_%M_%S"))

        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)
        # else:
        folder_to_save_checkpoints = folder_to_save + '/checkpoints'
        if not os.path.exists(folder_to_save_checkpoints):
            os.makedirs(folder_to_save_checkpoints)
        print('folder to save %s' % folder_to_save)
    else:
        folder_to_save = 'training/' + opt.continue_learning
        folder_to_save_checkpoints = folder_to_save + '/checkpoints'

    if opt.mask_in_loss or opt.masked_netE:
        plt.imshow(rectangle_mask[0,0,:,:].cpu())
        plt.savefig('%s/mask.jpg' % folder_to_save)

    # tensorboard
    writer = SummaryWriter(log_dir='%s/runs/%s' % (folder_to_save,os.path.basename(folder_to_save)))

    # load the generator
    if 'pgan' in opt.GAN:
        nets = networks.define_nets('proggan', opt.netG, use_RGBM=opt.masked,
                                    use_VAE=opt.vae_like, ckpt_path=None,
                                    load_encoder=False, device=device)
    elif 'BigGAN' in opt.GAN:
        nets = networks.define_nets('BigGAN', 'BigGAN', use_RGBM=opt.masked,
                                    use_VAE=opt.vae_like, ckpt_path=None,
                                    load_encoder=False, device=device,resolution=resolution)
    else:
        print('Error!')

    netG = nets.generator.eval()
    netG = netG.to(device)
    util.set_requires_grad(False, netG)
    # print(netG)

    # get latent and output shape
    out_shape = nets.setting['outdim']
    nz = nets.setting['nz']

    # create the encoder architecture
    depth = int(opt.netE_type.split('-')[-1])
    has_masked_input = opt.masked or opt.vae_like
    assert(not (opt.masked and opt.vae_like)), "specify 1 of masked or vae_like"
    which_model = 'orig'
    if opt.padding == 0:
        which_model = 'no_padd'
    elif 'unet' in opt.scenario_name:
        which_model = 'unet'
    elif 'insert_class' in opt.scenario_name:
        which_model = 'insert_class'
    if opt.predict_class:
        if opt.predict_class == 1:
            class_predicted = []
            class_orig = []
            nz_update = nz + 1
        elif opt.predict_class == 2:
            nz_update = nz + 1000
        elif opt.predict_class == 3 or opt.predict_class == 4:
            #c_shared
            nz_update = nz + 128
    else:
        nz_update = nz
    predicted_classes = [0]
    netE = proggan_networks.load_proggan_encoder(domain=None, nz=nz_update,
                                                 outdim=out_shape,
                                                 use_RGBM=opt.masked,
                                                 use_VAE=opt.vae_like,
                                                 resnet_depth=depth,
                                                 ckpt_path=None,which_model=which_model)

    if opt.masked_netE:
        netE = proggan_networks.MaskednetE(ratio,mask_width,netE)

    netE = netE.to(device).train()
    nets.encoder = netE
    # print(netE)
    debug = False
    if debug:
        netE = netE.to(device).eval()
        size_img = 512
        pixel = int(size_img/2)
        pulse_img = torch.zeros((1,3,size_img,size_img))
        pulse_response = torch.zeros(size_img-resolution,size_img-resolution)
        pulse_img[:,:,pixel,pixel] = 255
        for j in range(size_img-resolution):
            for i in range(size_img-resolution):
                tmp_img = pulse_img[:,:,j:j+resolution,i:i+resolution].to(device)
                encoded = netE(tmp_img, rectangle_mask, mask_width, opt.GAN)
                # if tmp_img.sum() == 0 and encoded.mean().detach() != 0:
                #     encoded = netE(tmp_img, rectangle_mask, mask_width, opt.GAN)
                # if (tmp_img*rectangle_mask).sum() != 0 and encoded.mean().detach() == 0:
                #     encoded = netE(tmp_img, rectangle_mask, mask_width, opt.GAN)
                pulse_response[j,i] = encoded.mean().detach()
    # losses + optimizers
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    perceptual_loss = losses.LPIPS_Loss(net='vgg', use_gpu=has_cuda).to(device)
    util.set_requires_grad(False, perceptual_loss)
    # resize img to 256 before lpips computation
    reshape = training_utils.make_ipol_layer(256)
    optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    start_ep = 0
    best_val_loss = float('inf')
  # z datasets
    train_loader = training_utils.training_loader(nets, batch_size, opt.seed)
    test_loader = training_utils.testing_loader(nets, batch_size, opt.seed)



    # load data from checkpoint
    if len(glob.glob(folder_to_save_checkpoints + '/netE*')) > 0 :
        opt.netE = np.sort(glob.glob(folder_to_save_checkpoints + '/netE*'))[-2]
    assert(not (opt.netE and opt.finetune)), "specify 1 of netE or finetune"
    if opt.finetune:
        checkpoint = torch.load(opt.finetune)
        sd = checkpoint['state_dict']
        # skip weights with dim mismatch, e.g. if finetuning from
        # an RGB encoder
        if sd['conv1.weight'].shape[1] != input_dim:
            # skip first conv if needed
            print("skipping initial conv")
            sd = {k: v for k, v in sd.items() if k != 'conv1.weight'}
        if sd['fc.bias'].shape[0] != nz:
            # skip fc if needed
            print("skipping fc layers")
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        netE.load_state_dict(sd, strict=False)
    if opt.netE:
        checkpoint = torch.load(opt.netE,map_location=device)
        netE.load_state_dict(checkpoint['state_dict'])
        optimizerE.load_state_dict(checkpoint['optimizer'])
        start_ep = checkpoint['epoch'] + 1
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']

        if opt.continue_learning:
            start_ep = 0
            folder_to_save += '/continue_learning'
            if not os.path.exists(folder_to_save):
                os.makedirs(folder_to_save)
            folder_to_save_checkpoints = folder_to_save + '/checkpoints'
            if not os.path.exists(folder_to_save_checkpoints):
                os.makedirs(folder_to_save_checkpoints)
            # opt.lambda_z_norm = 0.1
            print('continue learning with different params')
        else:
            print('continue learning from epoch %d' % start_ep)
            import shutil
            now = datetime.now()
            file_to_copy = '%s/train_loss.jpg' % (folder_to_save)
            file_new = '%s/train_loss_%s.jpg' % (folder_to_save,now.strftime("%d_%m_%Y_%H_%M_%S"))
            shutil.copy(file_to_copy, file_new)
            file_to_copy = '%s/z_norm.jpg' % (folder_to_save)
            file_new = '%s/z_norm_%s.jpg' % (folder_to_save,now.strftime("%d_%m_%Y_%H_%M_%S"))
            shutil.copy(file_to_copy, file_new)
            if opt.predict_class:
                file_to_copy = '%s/c_norm.jpg' % (folder_to_save)
                file_new = '%s/c_norm_%s.jpg' % (folder_to_save, now.strftime("%d_%m_%Y_%H_%M_%S"))
                shutil.copy(file_to_copy, file_new)
                # if opt.predict_class_one_hot:
                #     file_to_copy = '%s/class_predicted.jpg' % (folder_to_save)
                #     file_new = '%s/class_predicted_%s.jpg' % (folder_to_save, now.strftime("%d_%m_%Y_%H_%M_%S"))
                #     shutil.copy(file_to_copy, file_new)
    # uses 1600 samples per epoch, computes number of batches
    # based on batch size
    epoch_batches = 1600 // batch_size

    # eval the model with real images
    eval_model = False
    if eval_model:
        from skimage import io as img
        import copy
        #change FC to conv layer:
        netE_copy = copy.deepcopy(netE).eval()

        if opt.stride == 1:
            count = 0
            for name, module in netE._modules.items():
                # print(module)
                if 'conv' in name or 'max' in name:
                    module.dilation = (2 ** count, 2 ** count)
                    if module.stride != (1, 1):
                        count += 1
                        module.stride = (1, 1)
                if 'layer' in name:
                    for j in range(len(module)):
                        for name_layer, module_layer in module[j]._modules.items():
                            if 'conv' in name_layer or 'max' in name_layer:
                                module_layer.dilation = (2 ** count, 2 ** count)
                                if module_layer.stride != (1, 1):
                                    count += 1
                                    module_layer.stride = (1, 1)
                            if j == 0 and 'downsample' in name_layer:
                                module_layer[0].dilation = (2 ** count, 2 ** count)
                                if module_layer[0].stride != (1, 1):
                                    # count += 1
                                    module_layer[0].stride = (1, 1)

            in_ch = 512
            out_ch = 512
            kernel_size = 2
            netE.avgpool = nn.Conv2d(in_ch,out_ch,kernel_size,stride=1,padding=0,dilation=(2 ** count,2 ** count),bias=False).to(device)
            netE.avgpool.weight.data.fill_(1 / (kernel_size ** 2))

            fc = netE.fc.state_dict()
            in_ch = fc["weight"].size(1)
            out_ch = fc["weight"].size(0)
            conv = nn.Conv2d(in_ch, out_ch, 1, dilation = (2 ** count))
            conv.load_state_dict({"weight": fc["weight"].view(out_ch, in_ch, 1, 1),
                                  "bias": fc["bias"]})
            netE.fc = conv.to(device)

        print(netE)

        # load natural images
        folder = '../SinGAN-master/Input/for_vgg/real'
        files = glob.glob('%s/*' % folder)
        real_img = torch.zeros((len(files), 3, resolution, resolution))
        for i, file in enumerate(files):
            if ('png' in file or 'jpg' in file):
                x = img.imread(file)
                if x.shape[0] < resolution or x.shape[1] < resolution:
                    continue
                    # x = cv2.resize(x, (resolution, resolution))
                # plt.imshow(x)
                # plt.show()
                x = x[:, :, :, None]
                x = x.transpose((3, 2, 0, 1)) / 255
                x = ((x - 0.5) * 2)  # .clamp(-1,1)
                x = torch.from_numpy(x).to(device)
                print(x.dtype)
                x = x.type(torch.cuda.FloatTensor) if (has_cuda) else x.type(torch.FloatTensor)
                # output = training_utils.best_place_to_insert(x, netE, resolution, folder_to_save,device)
                real_img[i, :, :, :] = x[:,:,0:resolution, 0:resolution]
                # real_img[i+1, :, :, :] = x[:, :, 1:1+resolution, 0:resolution]
                # real_img[i + 2, :, :, :] = x[:, :, 1:1 + resolution, 1:1+resolution]
                # real_img[i + 3, :, :, :] = x[:, :, 0:resolution, 1:1+resolution]
                # real_img = torch.zeros((1, 3, x.shape[2], x.shape[3]))
                # real_img_tmp = x[:,:,0:resolution, 0:resolution]
                # output = netE(real_img_tmp.to(device))
                # real_img_tmp = x[:,:,0:resolution, 0:resolution]
                # output = netE_copy(real_img_tmp.to(device))
                # break
                #
        # pass images through netE:
        num_itr = int(np.ceil(len(files) / batch_size))
        real_img = real_img.to(device)
        folder_eval = os.path.join(folder_to_save, 'eval')
        if not os.path.exists(folder_eval):
            os.makedirs(folder_eval)
        for j in range(num_itr):
            if 4 * j + 4 < real_img.shape[0]:
                real_img_tmp = real_img[4 * j:4 * j + 4, :, :, :]
            else:
                real_img_tmp = real_img[-1 - batch_size:-1, :, :, :]
            if opt.masked_netE:
                encoded = netE_copy(real_img_tmp, rectangle_mask, mask_width, opt.GAN)
            else:
                encoded = netE_copy(real_img_tmp)
            encoded = encoded.reshape([batch_size, encoded.shape[1]])
            c = np.ones((batch_size,)) * class_number
            category = torch.Tensor([c]).long().to(device)
            c_shared = netG.shared(category).to(device)[0]
            regenerated = netG(encoded, c_shared)
            # if opt.masked_netE or opt.mask_in_loss:
            display_results(opt, resolution, real_img_tmp, regenerated, folder_eval, 0, j, mask_width)
            # else:
            #     display_results(opt, resolution, real_img_tmp, regenerated, folder_eval, 0, j)
        netE = netE.train()

    #save opt arguments
    file_name = folder_to_save + r'/inf.txt'
    opt.file_name = file_name
    f = open(file_name,"a")
    now = datetime.now()
    f.write(str(now.strftime("%d/%m/%Y %H:%M:%S")))
    f.write("\n")
    f.write(folder_to_save)
    f.write("\n")
    f.write(str(opt))
    f.write("\n")
    f.write('Encoder:')
    f.write("\n")
    f.write(str(netE))
    f.write("\n")
    f.write(str('Generator'))
    f.write("\n")
    f.write(str(netG))
    f.write("\n")
    f.write(str(class_number))
    f.write("\n")
    f.close()

    total_loss = []
    mse_total_loss = []
    z_total_loss = []
    z_norm_total = []
    encoded_norm_total = []
    l1_total_loss = []
    c_norm_total = []
    encoded_c_total = []
    perceptual_total_loss = []
    norm_z_loss_total = []
    c_loss_total = []
    norm_c_total_loss = []
    for epoch, epoch_loader in enumerate(pbar(
        training_utils.epoch_grouper(train_loader, epoch_batches),
        total=(opt.niter-start_ep)), start_ep):

        # stopping condition
        if epoch > opt.niter:
            break
        # run a train epoch of epoch_batches batches
        for step, z_batch in enumerate(pbar(
            epoch_loader, total=epoch_batches), 1):
            # if opt.predict_class and step > 50:
            #     opt.lambda_c = 10
            if 'pgan' in opt.GAN:
                z_batch = z_batch.to(device)
                fake_im = netG(z_batch).detach()
            elif 'BigGAN' in opt.GAN:
                z_batch = torch.normal(0, 1, size=[batch_size, nz]).to(device)
                c = class_number[np.random.randint(low=0, high=opt.number_of_classes, size=(batch_size,))]
                # c = 208 * np.ones((15,))
                # c = np.random.randint(low=0, high=999, size=(15,)) ##
                # c[0] = 208 ##
                category = torch.Tensor([c]).long().to(device)
                if opt.predict_class == 1:
                    class_orig.append(category[0].cpu().numpy())
                c_shared = netG.shared(category).to(device)[0]
                # z_batch_new = z_batch[0,:] * torch.ones([15,120]).cuda() ##
                # fake_im_new = netG(z_batch_new, c_shared).detach()
                # plt.figure(figsize=(20,10))
                # z_batch_new = z_batch[1, :] * torch.ones([15, 120]).cuda()
                # stds = np.linspace(0,1,15)
                # fake_im_new = netG(z_batch_new, c_shared).detach()
                # for i in range(1,15):
                #     c_shared[i,:] += torch.normal(0,stds[i],[128]).to(device)
                #
                # fake_im_new = netG(z_batch_new, c_shared).detach()
                # for i in range(15):
                #     plt.subplot(3,5,i+1)
                #     plt.imshow(convert_image_np(fake_im_new[i, :, :, :].reshape((1, 3, resolution, resolution))))
                #     plt.title('std=%f' % stds[i])
                #     plt.axis('off')
                # plt.savefig('same_z_and_class_with_noise_std_max_1_208_a.jpg')
                # plt.show() ##
                fake_im = netG(z_batch, c_shared).detach()

            netE.zero_grad()
            if has_masked_input:
                hints_fake, mask_fake = masking.mask_upsample(fake_im)
                # mask_fake = rectangle_mask - 0.5
                # hints_fake = fake_im * rectangle_mask
                encoded = netE(torch.cat([hints_fake, mask_fake], dim=1))
                if opt.masked:
                    if 'BigGAN' in opt.GAN:
                        encoded = encoded.reshape([batch_size, encoded.shape[1]])
                        regenerated = netG(encoded, c_shared)
                    else:
                        regenerated = netG(encoded)
                elif opt.vae_like:
                    sample = torch.randn_like(encoded[:, nz:, :, :])
                    encoded_mean = encoded[:, nz:, :, :]
                    encoded_sigma = torch.exp(encoded[:, :nz, :, :])
                    reparam = encoded_mean + encoded_sigma * sample
                    regenerated = netG(reparam)
                    encoded = encoded_mean # just use mean in z loss
            else:
                # standard RGB encoding
                if opt.masked_netE:
                    # fake_im[:, :, mask_width:-mask_width, mask_width:-mask_width] = np.nan
                    encoded = netE(fake_im,rectangle_mask,mask_width,opt.GAN)
                    if debug:
                        gradients = torch.autograd.grad(outputs=encoded, inputs=fake_im,
                                                        grad_outputs=torch.ones_like(encoded).to(device),
                                                        # .cuda(), #if use_cuda else torch.ones(
                                                        # disc_interpolates.size()),
                                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
                else:
                    if 'insert_class' in which_model:
                        encoded = netE(fake_im,c_shared)
                    else:
                        encoded = netE(fake_im)
                if 'BigGAN' in opt.GAN:
                    encoded = encoded.reshape([batch_size, encoded.shape[1]])
                    if opt.predict_class:
                        encoded_c = encoded[:,120::]
                        encoded = encoded[:,0:120]
                        if opt.predict_class == 3 or opt.predict_class == 4:
                            c_shared_encoded = torch.sigmoid(encoded_c) - 0.5 #most vec in W are between -0.5 - 0.5
                        elif opt.predict_class == 1:
                            encoded_c = torch.sigmoid(encoded_c)
                            category = (1000*encoded_c).long().t()
                            predicted_classes = category[0]
                            class_predicted.append(category[0].cpu().numpy())
                            c_shared_encoded = netG.shared(category).to(device)[0]
                        elif opt.predict_class == 2:
                            encoded_c == torch.sigmoid(encoded_c)
                            # idx = torch.argmax(encoded_c,axis=1)
                            # c_shared_encoded = netG.shared(idx).to(device)
                            c_shared_encoded = torch.matmul(netG.shared.weight.t(),encoded_c.t()).t()

                        regenerated = netG(encoded, c_shared_encoded)
                    else:
                        regenerated = netG(encoded, c_shared)
                else:
                    regenerated = netG(encoded)
            text = "Epoch %d step %d losses:" % (epoch, step)
            # compute loss
            loss = 0
            if 'MSE' in opt.losses:
                if opt.mask_in_loss:
                    loss_mse = torch.mean((regenerated - fake_im) ** 2 * rectangle_mask) * pixel_ratio
                else:
                    loss_mse = mse_loss(regenerated, fake_im)
                loss_mse *= opt.lambda_mse
                mse_total_loss.append(loss_mse.detach().cpu().numpy())
                text += ' mse %0.4f' % loss_mse.item()
                loss += loss_mse
            if 'L1' in opt.losses:
                loss_l1 = l1_loss(regenerated,fake_im)
                l1_total_loss.append(loss_l1.detach().cpu().numpy())
                loss += loss_l1
            if 'Z' in opt.losses:
                loss_z = opt.lambda_latent * training_utils.cor_square_error_loss(encoded, z_batch)
                z_total_loss.append(loss_z.detach().cpu().numpy())
                text += ' z %0.4f' % loss_z.item()
                loss += loss_z
            if 'PERCEPTUAL' in opt.losses:
                if opt.small_RF_lpips:
                    if opt.DEBUG_PERCEPTUAL:
                        regenerated_vgg = vgg_output(regenerated, opt.vggModel,stop_layer=8)
                        fake_vgg = vgg_output(fake_im, opt.vggModel,stop_layer=8)
                        diff = torch.zeros((len(regenerated_vgg)))
                        for j in range(len(regenerated_vgg)):
                            diff[j] = ((regenerated_vgg[j] - fake_vgg[j]) ** 2).mean()
                        loss_perceptual = opt.lambda_lpips * diff.mean()
                    else:
                        # tmp = (perceptual_loss.forward(reshape(regenerated), reshape(fake_im),stop_layer=2))
                        if opt.mask_width < 64:
                            stop_slice = 2
                        else:
                            stop_slice = 3
                        if opt.mask_in_loss:
                            loss_perceptual = opt.lambda_lpips * (perceptual_loss.forward(
                                reshape(regenerated), reshape(fake_im),stop_layer=[stop_slice,True,opt.mask_width])).mean()
                        else:
                            loss_perceptual = opt.lambda_lpips * (perceptual_loss.forward(
                                reshape(regenerated), reshape(fake_im),stop_layer=[stop_slice,None])).mean() #receptive field of 14*14
                else:
                    if opt.DEBUG_PERCEPTUAL:
                        regenerated_vgg = vgg_output(regenerated, opt.vggModel)
                        fake_vgg = vgg_output(fake_im, opt.vggModel)
                        diff = torch.zeros((len(regenerated_vgg)))
                        for j in range(len(regenerated_vgg)):
                            diff[j] = ((regenerated_vgg[j] - fake_vgg[j]) ** 2).mean()
                        loss_perceptual = opt.lambda_lpips * diff.mean()
                    else:
                        loss_perceptual = opt.lambda_lpips * perceptual_loss.forward(
                            reshape(regenerated), reshape(fake_im)).mean()
                perceptual_total_loss.append(loss_perceptual.detach().cpu().numpy())
                text += ' lpips %0.4f' % loss_perceptual.item()
                loss += loss_perceptual
            if ('norm0' in opt.losses or 'norm1' in opt.losses):
                if 'norm0' in opt.losses:
                    norm_z_loss = opt.lambda_z_norm * (encoded ** 2).mean()
                                  # mse_loss(encoded.norm(2,dim=1),z_batch.norm(2,dim=1))
                elif 'norm1' in opt.losses:
                    norm_z_loss = opt.lambda_z_norm * ((encoded ** 2).mean() - 1)**2
                norm_z_loss_total.append(norm_z_loss.detach().cpu())
                text += ' norm z %0.4f' % norm_z_loss.item()
                loss += norm_z_loss
            if opt.predict_class == 2:
                encoded_c_sorted = torch.sort(encoded_c, descending=True, axis=1)[0]
                predicted_classes = torch.argmax(encoded_c,axis=1)
                loss_one_hot = (encoded_c_sorted[:,0] - 1).abs().mean() + (encoded_c_sorted[:,1::]).abs().mean(axis=1).sum()
                loss_one_hot = opt.lambda_c * loss_one_hot
                # print(encoded_c_sorted[:,0] , (encoded_c_sorted[:,1::]).abs().mean())
                c_loss_total.append(loss_one_hot.detach().cpu().numpy())
                loss += loss_one_hot
                #one hot vec. set the biggest one to be close to 1 and the others to 0
            if opt.predict_class == 4:
                # import torch.nn.functional as F
                # similarity = F.cosine_similarity(encoded_c.unsqueeze(1), netG.shared.weight, dim=-1)
                #calc mse between vecs
                predicted_classes = torch.zeros((batch_size))
                loss_c = 0
                for j in range(batch_size):
                    tmp = torch.argmin(((encoded_c[j, :] - netG.shared.weight) ** 2).sum(axis=1))
                    loss_c += ((encoded_c[j, :] - netG.shared.weight[tmp,:]) ** 2).mean()
                    predicted_classes[j] = tmp
                loss_c = opt.lambda_c * loss_c
                c_loss_total.append(loss_c.detach().cpu().numpy())
                loss += loss_c
                #nearest neighbor
            if opt.predict_class:
                c_norm_total.append((c_shared ** 2).mean().cpu())
                encoded_c_total.append((c_shared_encoded ** 2).mean().detach().cpu())
            if 'normC' in opt.losses:
                norm_C_loss = opt.lambda_c_norm * (c_shared_encoded ** 2).mean()
                norm_c_total_loss.append(norm_C_loss.detach().cpu().numpy())
                loss +=  norm_C_loss
            # optimize
            text += ' total loss %0.4f' % loss.item()
            loss.backward()
            optimizerE.step()
            # if 'norm' in opt.losses:
            z_norm_total.append((z_batch ** 2).mean().cpu())
            encoded_norm_total.append((encoded ** 2).mean().detach().cpu())

            total_loss.append(loss.detach().cpu().numpy())
            # send losses to tensorboard
            if (epoch % 20 ==0 or epoch % 201==0) and step % 20 == 0:
                total_batches = epoch * epoch_batches + step
                if 'Z' in opt.losses:
                    writer.add_scalar('%s/loss/train_z' % folder_to_save, loss_z, total_batches)
                if 'MSE' in opt.losses:
                    writer.add_scalar('%s/loss/train_mse' % folder_to_save, loss_mse, total_batches)
                if 'PERCEPTUAL' in opt.losses:
                    writer.add_scalar('%s/loss/train_lpips' % folder_to_save, loss_perceptual, total_batches)
                writer.add_scalar('%s/loss/train_total' % folder_to_save , loss, total_batches)
                # pbar.print("Epoch %d step %d Losses z %0.4f mse %0.4f lpips %0.4f total %0.4f"
                #            % (epoch, step, loss_z.item(), loss_mse.item(),
                #               loss_perceptual.item(), loss.item()))

                f = open(opt.file_name, "a")
                f.write(text)
                f.write("\n")
                f.close()

                x = np.linspace(0,len(total_loss)-1,len(total_loss))
                legends = []
                if 'MSE' in opt.losses:
                    plt.plot(x,mse_total_loss)
                    legends.append('%.3f MSE'% opt.lambda_mse)
                if 'Z' in opt.losses:
                    plt.plot(x,z_total_loss)
                    legends.append('%.3f Z'% opt.lambda_latent)
                if 'PERCEPTUAL' in opt.losses:
                    plt.plot(x,perceptual_total_loss)
                    legends.append('%.3f PERCEPTUAL'% opt.lambda_lpips)
                if ('norm0' in opt.losses or 'norm1' in opt.losses):
                    plt.plot(x,norm_z_loss_total)
                    legends.append('%.3f norm_Z_diff' % opt.lambda_z_norm)
                if opt.predict_class == 2 or opt.predict_class == 4:
                    plt.plot(x,c_loss_total)
                    legends.append('%.3f c_loss' % opt.lambda_c)
                if 'normC' in opt.losses:
                    plt.plot(x,norm_c_total_loss)
                    legends.append('%.3f norm_c_loss' % opt.lambda_c_norm)
                if 'L1' in opt.losses:
                    plt.plot(x,l1_total_loss)
                    legends.append('L1')
                plt.plot(x, total_loss)
                legends.append('total loss')
                plt.legend(legends)

                plt.savefig('%s/train_loss.jpg' % (folder_to_save))
                plt.close()
                # if 'norm' in opt.losses:
                plt.figure()
                plt.plot(x,z_norm_total)
                plt.plot(x,encoded_norm_total)
                plt.legend(['z_norm','encoded_norm'])
                plt.savefig('%s/z_norm.jpg' % (folder_to_save))
                plt.close()

                if opt.predict_class:
                    plt.figure()
                    plt.plot(x, c_norm_total)
                    plt.plot(x, encoded_c_total)
                    plt.legend(['c_norm', 'encoded_norm'])
                    plt.savefig('%s/c_norm.jpg' % (folder_to_save))
                    plt.close()
                    if opt.predict_class == 1:
                        plt.figure()
                        class_orig_t = np.asarray(class_orig).reshape(-1)
                        class_predicted_t = np.asarray(class_predicted).reshape(-1)
                        x_t = np.linspace(0,len(class_orig_t)-1,len(class_orig_t))
                        plt.plot(x_t, class_orig_t)
                        plt.plot(x_t, class_predicted_t)
                        plt.legend(['c_orig', 'c_predicted'])
                        plt.savefig('%s/class_predicted.jpg' % (folder_to_save))
                        plt.close()

                if not opt.masked:
                    if opt.predict_class:
                        display_results(opt, resolution, fake_im, regenerated, folder_to_save, epoch, step,mask_width,predicted_classes)
                    else:
                        display_results(opt, resolution, fake_im, regenerated, folder_to_save, epoch, step, mask_width)
                else:
                    display_results(opt, resolution, fake_im, regenerated, folder_to_save, epoch, step,mask_width, hints_fake)

            if step == 1:
                total_batches = epoch * epoch_batches + step
                if has_masked_input:
                    grid = vutils.make_grid(
                        torch.cat((reshape(fake_im), reshape(hints_fake),
                                   reshape(regenerated))),
                        nrow=8, normalize=True, scale_each=(-1, 1))
                else:
                    grid = vutils.make_grid(
                        torch.cat((reshape(fake_im), reshape(regenerated))), nrow=8,
                        normalize=True, scale_each=(-1, 1))
                writer.add_image('Train Image', grid, total_batches)

        # print('finished training')

        # updated to run a small set of test zs
        # rather than a single fixed batch
        netE.eval()
        test_metrics = {
            'loss_z': util.AverageMeter('loss_z'),
            'loss_mse': util.AverageMeter('loss_mse'),
            'loss_perceptual': util.AverageMeter('loss_perceptual'),
            'loss_total': util.AverageMeter('loss_total'),
        }

        folder_to_save_test = folder_to_save + '/test'
        if not os.path.exists(folder_to_save_test):
            os.mkdir(folder_to_save_test)

        for step, test_zs in enumerate(pbar(test_loader), 1):
            with torch.no_grad():
                if 'BigGAN' in opt.GAN:
                    test_zs = torch.normal(0, 1, size=[batch_size, nz]).to(device)
                    c = class_number[np.random.randint(low=0, high=opt.number_of_classes, size=(batch_size,))]
                    category = torch.Tensor([c]).long().to(device)
                    c_shared = netG.shared(category).to(device)[0]
                    fake_im = netG(z_batch, c_shared)
                else:
                    test_zs = test_zs.to(device)
                    fake_im = netG(test_zs)
                if has_masked_input:
                    hints_fake, mask_fake = masking.mask_upsample(fake_im)
                    encoded = netE(torch.cat([hints_fake, mask_fake], dim=1))
                    if 'BigGAN' in opt.GAN:
                        encoded = encoded.reshape([batch_size, encoded.shape[1]])

                    if opt.masked:
                        if 'BigGAN' in opt.GAN:
                            regenerated = netG(encoded, c_shared)
                        else:
                         regenerated = netG(encoded)
                    elif opt.vae_like:
                        sample = torch.randn_like(encoded[:, nz:, :, :])
                        encoded_mean  = encoded[:, nz:, :, :]
                        encoded_sigma = torch.exp(encoded[:, :nz, :, :])
                        reparam = encoded_mean + encoded_sigma * sample
                        regenerated = netG(reparam)
                        encoded = encoded_mean # just use mean in z loss
                else:
                    if opt.masked_netE:
                        encoded = netE(fake_im, rectangle_mask,mask_width, opt.GAN)
                    else:
                        if 'insert_class' in which_model:
                            encoded = netE(fake_im, c_shared)
                        else:
                            encoded = netE(fake_im)

                    if 'BigGAN' in opt.GAN:
                        encoded = encoded.reshape([batch_size, encoded.shape[1]])
                        if opt.predict_class:
                            encoded_c = encoded[:, 120::]
                            encoded = encoded[:, 0:120]
                            if opt.predict_class == 1:
                                encoded_c = torch.sigmoid(encoded_c)
                                category = (1000 * encoded_c).long().t()
                                predicted_classes = category[0]
                                class_predicted.append(category[0].cpu().numpy())
                                c_shared_encoded = netG.shared(category).to(device)[0]
                            elif opt.predict_class == 3 or opt.predict_class == 4:
                                c_shared_encoded = torch.sigmoid(encoded_c) - 0.5  # most vec in W are between -0.5 - 0.5
                                # encoded_c = torch.sigmoid(encoded_c)
                                # category = (1000 * encoded_c).long().t()
                                # c_shared_encoded = netG.shared(category).to(device)[0]
                            elif opt.predict_class == 2:
                                encoded_c == torch.sigmoid(encoded_c)
                                c_shared_encoded = torch.matmul(netG.shared.weight.t(), encoded_c.t()).t()
                            regenerated = netG(encoded, c_shared_encoded)
                        else:
                            regenerated = netG(encoded, c_shared)
                    else:
                        regenerated = netG(encoded)

                # compute loss
                loss_z = training_utils.cor_square_error_loss(encoded, test_zs)
                loss_mse = mse_loss(regenerated, fake_im)
                loss_perceptual = perceptual_loss.forward(
                    reshape(regenerated), reshape(fake_im)).mean()

                loss = (opt.lambda_latent * loss_z + opt.lambda_mse * loss_mse
                        + opt.lambda_lpips * loss_perceptual)

                if epoch % 10 == 0 and step % 5 == 0:
                    if not opt.masked:
                        display_results(opt, resolution, fake_im, regenerated, folder_to_save_test, epoch, step, mask_width)
                    else:
                        display_results(opt, resolution, fake_im, regenerated, folder_to_save_test, epoch, step, mask_width, hints_fake)


            # update running avg
            test_metrics['loss_z'].update(loss_z)
            test_metrics['loss_mse'].update(loss_mse)
            test_metrics['loss_perceptual'].update(loss_perceptual)
            test_metrics['loss_total'].update(loss)

            # save a fixed batch for visualization
            if step == 1:
                if has_masked_input:
                    grid = vutils.make_grid(
                        torch.cat((reshape(fake_im), reshape(hints_fake),
                                   reshape(regenerated))),
                        nrow=8, normalize=True, scale_each=(-1, 1))
                else:
                    grid = vutils.make_grid(
                        torch.cat((reshape(fake_im), reshape(regenerated))), nrow=8,
                        normalize=True, scale_each=(-1, 1))

        # send to tensorboard
        writer.add_scalar('%s/loss/test_z' % folder_to_save_test, test_metrics['loss_z'].avg, epoch)
        writer.add_scalar('%s/loss/test_mse' % folder_to_save_test, test_metrics['loss_mse'].avg, epoch)
        writer.add_scalar('%s/loss/test_lpips' % folder_to_save_test, test_metrics['loss_perceptual'].avg, epoch)
        writer.add_scalar('%s/loss/test_total' % folder_to_save_test, test_metrics['loss_total'].avg, epoch)
        writer.add_image('Test Image', grid, epoch)
        netE.train()

        # do checkpointing
        if epoch % 20 == 0 or epoch == opt.niter:
            training_utils.make_checkpoint(
                netE, optimizerE, epoch,
                test_metrics['loss_total'].avg.item(),
                '%s/netE_epoch_%d.pth' % (folder_to_save_checkpoints, epoch))
        # if epoch == opt.niter:
        #     cmd = 'ln -s netE_epoch_%d.pth %s/model_final.pth' % (epoch, opt.outf)
        #     os.system(cmd)
        if test_metrics['loss_total'].avg.item() < best_val_loss:
            # modified to save based on test zs loss rather than
            # final model at the end
            pbar.print("Checkpointing at epoch %d" % epoch)
            training_utils.make_checkpoint(
                netE, optimizerE, epoch,
                test_metrics['loss_total'].avg.item(),
                '%s/netE_epoch_best.pth' % (folder_to_save_checkpoints))
            best_val_loss = test_metrics['loss_total'].avg.item()
        try:
            # eval the model with real images
            eval_model = False
            if eval_model:
                netE = netE.eval()
                # load natural images
                folder = '../SinGAN-master/Input/for_vgg/real'
                files = glob.glob('%s/*' % folder)
                real_img = torch.zeros((len(files), 3, resolution, resolution))
                for i, file in enumerate(files):
                    if ('png' in file or 'jpg' in file):
                        x = img.imread(file)
                        if x.shape[0] < resolution or x.shape[1] < resolution:
                            continue
                            # x = cv2.resize(x, (resolution, resolution))
                        else:
                            training_utils.best_place_to_insert(x, netE, resolution, folder_to_save)
                            x = x[0:resolution, 0:resolution, :]
                        # plt.imshow(x)
                        # plt.show()
                        x = x[:, :, :, None]
                        x = x.transpose((3, 2, 0, 1)) / 255
                        x = ((x - 0.5) * 2)  # .clamp(-1,1)
                        x = torch.from_numpy(x)

                        real_img[i, :, :, :] = x
                        # x = x.type(torch.cuda.FloatTensor) if not (opt.not_cuda) else x.type(torch.FloatTensor)
                # pass images through netE:
                num_itr = int(np.ceil(len(files) / batch_size))
                real_img = real_img.to(device)
                folder_eval = os.path.join(folder_to_save, 'eval')
                if not os.path.exists(folder_eval):
                    os.makedirs(folder_eval)
                for j in range(num_itr):
                    if 4 * j + 4 < real_img.shape[0]:
                        real_img_tmp = real_img[4 * j:4 * j + 4, :, :, :]
                    else:
                        real_img_tmp = real_img[-1 - batch_size:-1, :, :, :]
                    if opt.masked_netE:
                        encoded = netE(real_img_tmp, rectangle_mask, mask_width, opt.GAN)
                    else:
                        encoded = netE(real_img_tmp)
                    encoded = encoded.reshape([batch_size, 120])
                    c = np.ones((batch_size,)) * class_number
                    category = torch.Tensor([c]).long().to(device)
                    c_shared = netG.shared(category).to(device)[0]
                    regenerated = netG(encoded, c_shared)
                    # if opt.masked_netE or opt.mask_in_loss:
                    display_results(opt, resolution, real_img_tmp, regenerated, folder_eval, 0, j, mask_width)
                    # else:
                    #     display_results(opt, resolution, real_img_tmp, regenerated, folder_eval, 0, j)
                netE = netE.train()
        except:
            print('failed in eval')

if __name__ == '__main__':
    import time
    t = time.time()
    parser = training_utils.make_parser()
    opt = parser.parse_args()
    print(opt)

    opt.outf = opt.outf.format(**vars(opt))

    os.makedirs(opt.outf, exist_ok=True)
    # save options
    with open(os.path.join(opt.outf, 'optE.yml'), 'w') as f:
        yaml.dump(vars(opt), f, default_flow_style=False)
    # torch.autograd.set_detect_anomaly(True)
    train(opt)
    elapsed = time.time() - t
    print(elapsed)
