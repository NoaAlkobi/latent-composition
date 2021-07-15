import torch
import argparse
import itertools

### arguments

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--netE_type', type=str, default='resnet-18', required=False, help='type of encoder architecture; e.g. resnet-18, resnet-34')
    parser.add_argument('--netG', type=str, required=False, default='church', help="generator to load")
    parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
    parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--outf', default='.', help='folder to output model checkpoints')
    parser.add_argument('--seed', default=0, type=int, help='manual seed')
    parser.add_argument('--gpu_num', default=0, type=int, help='which gpu to send to')
    parser.add_argument('--lambda_latent', default=1.0, type=float, help='loss weighting (latent recovery)')
    parser.add_argument('--lambda_mse', default=1.0, type=float, help='loss weighting (image mse)')
    parser.add_argument('--lambda_z_norm', default=0.01, type=float, help='loss weighting (image mse)')
    parser.add_argument('--lambda_lpips', default=1.0, type=float, help='loss weighting (image perceptual)')
    parser.add_argument('--lambda_id', default=0.0, type=float, help='loss weighting (optional identity loss for faces)')
    parser.add_argument('--netE', default='', help="path to netE (to continue training)")
    parser.add_argument('--finetune', type=str, default='', help="finetune from weights at this path")
    parser.add_argument('--masked', action='store_true', help="train with masking")
    parser.add_argument('--vae_like', action='store_true', help='train with masking, predict mean and sigma (not used in paper)')
    parser.add_argument('--GAN', type=str, default='BigGAN') #BigGAN
    parser.add_argument('--mask_in_loss', type=int, default=0)
    parser.add_argument('--masked_netE', type=int, default=0)
    parser.add_argument('--small_RF_lpips', type=int, default=0)
    parser.add_argument('--continue_learning', type=str, default='')
    parser.add_argument('--DEBUG_PERCEPTUAL', type=int, default=0)
    parser.add_argument('--one_class_only', type=int, default=0)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--mask_width', type=int, default=16)
    parser.add_argument('--losses', type=str, default='MSE_PERCEPTUAL_Z')
    parser.add_argument('--scenario_name', type=str, default='')
    parser.add_argument('--padding', default=1, type=int, help='padding resnet')
    parser.add_argument('--stride', default=0, type=int, help='stride resnet')
    # parser.add_argument('--only_mse', type=int, default=0)
    # parser.add_argument('--orig_code', type=int, default=1)

    return parser

### checkpointing

def make_checkpoint(netE, optimizer, epoch, val_loss, save_path):
    sd = {
        'state_dict': netE.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }
    torch.save(sd, save_path)

### loss functions

def cor_square_error_loss(x, y, eps=1e-8):
    # Analogous to MSE, but in terms of Pearson's correlation
    return (1.0 - torch.nn.functional.cosine_similarity(x, y, eps=eps)).mean()

### interpolation utilities

def make_ipol_layer(size):
    return torch.nn.AdaptiveAvgPool2d((size, size))
    # return InterpolationLayer(size)

class InterpolationLayer(torch.nn.Module):
    def __init__(self, size):
        super(InterpolationLayer, self).__init__()
        self.size=size

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x, size=self.size, mode='area')


### dataset utilities

def training_loader(nets, batch_size, global_seed=0):
    '''
    Returns an infinite generator that runs through randomized z
    batches, forever.
    '''
    g_epoch = 1
    while True:
        z_data = nets.sample_zs(n=10000, seed=g_epoch+global_seed,
                                device='cpu')
        dataloader = torch.utils.data.DataLoader(
                z_data,
                shuffle=False,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True)
        for batch in dataloader:
            yield batch
        g_epoch += 1

def testing_loader(nets, batch_size, global_seed=0,GAN='pgan'):
    '''
    Returns an a short iterator that returns a small set of test data.
    '''
    z_data = nets.sample_zs(n=10*batch_size, seed=global_seed,
                            device='cpu')
    dataloader = torch.utils.data.DataLoader(
            z_data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True)
    return dataloader

def epoch_grouper(loader, epoch_size, num_epochs=None):
    '''
    To use with the infinite training loader: groups the training data
    batches into epochs of the given size.
    '''
    it = iter(loader)
    epoch = 0
    while True:
        chunk_it = itertools.islice(it, epoch_size)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)
        epoch += 1
        if num_epochs is not None and epoch >= num_epochs:
            return

def best_place_to_insert(img,netE,resolution,folder_to_save,device):
    import os
    import numpy as np
    import cv2
    rectangle_size = resolution
    print(img.shape)
    img = img.to(device)
    output = netE(img)


    os.makedirs(folder_to_save,exist_ok=True)
    fullyConvolutional = True
    score_sorted, idx_class = torch.sort(vec, descending=True,dim=1)
    idx_class = idx_class.cpu().numpy()[0]
    score_sorted = score_sorted.detach().cpu().numpy()[0]
    highest_class = idx_class[0]

    if fullyConvolutional:
        fig, ax = plt.subplots(1,2, figsize=(25,10))
        ax[0].imshow(functions.convert_image_np(img))
        ax[0].set_title('img_shape_%d_%d' % (img.shape[2],img.shape[3]))
        data = idx_class[0,:,:]
        classes,amount = np.unique(data, return_counts=True)
        # idx = np.argsort(amount)
        # classes = classes[idx]
        # amount = amount[idx]
        colors = ["blue","green","red","cyan","magenta","yellow","black","white","orange","pink",]
        if len(classes) > len(colors):
            for p in range(len(classes)-len(colors)):
                colors.append([randrange(255) / 255, randrange(255) / 255, randrange(255) / 255])
        cm = ListedColormap(colors[0:len(classes)])
        # classes, amount = np.unique(data, return_counts=True)
        percentage = 100 * (amount / amount.sum())
        labels = []
        labels_only = ''
        for p in range(len(classes)):
            string = idx2label[classes[p]] + '_%.2f%%' % (percentage[p])
            labels.append(string)
            if p%3==0:
                labels_only += '\n'
            labels_only = labels_only + ' * %d_%s' %(classes[p], str(idx2label[classes[p]]))
            # labels_only.append(idx2label[classes[p]])
        # labels = np.array([idx2label[x] + '_%f' for x in vals])
        len_lab = len(labels)
        norm_bins = np.sort(classes) + 0.5
        norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
        # print(norm_bins)
        norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
        fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
        im = ax[1].imshow(data, cmap=cm, norm=norm)
        ax[1].set_title('%s' % (labels_only))
        diff = norm_bins[1:] - norm_bins[:-1]
        tickz = norm_bins[:-1] + diff / 2
        cb = fig.colorbar(im, format=fmt,cmap=cm, ticks=tickz)
        if normalize_img:
            plt.savefig('%s/%s_fully_connected_%.2f_factor_normalized_image.jpg' % (folder_save, img1[:-4],factor))
        else:
            plt.savefig('%s/%s_fully_connected_%.2f_factor.jpg' % (folder_save, img1[:-4],factor))
        plt.close()