from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import oyaml as yaml
from utils import pbar, util, masking, losses, training_utils
from networks import networks, proggan_networks, biggan
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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

def train(opt):
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if has_cuda else "cpu")
    batch_size = int(opt.batchSize)
    # batch_size = 4
    resolution = 256
    cudnn.benchmark = True
    rectangle_mask = torch.ones((batch_size,1,resolution,resolution)).to(device)
    mask_width = int(resolution / 10)
    rectangle_mask[:,:,mask_width+1:-mask_width,mask_width+1:-mask_width] = 0

    # tensorboard
    writer = SummaryWriter(log_dir='training/runs/%s' % os.path.basename(opt.outf))

    # load the generator
    nets = networks.define_nets('BigGAN', 'BigGAN', use_RGBM=opt.masked,
                                use_VAE=opt.vae_like, ckpt_path=None,
                                load_encoder=False, device=device,resolution=resolution)
    netG = nets.generator
    util.set_requires_grad(False, netG)
    netG.eval()
    print(netG)

    # get latent and output shape
    out_shape = nets.setting['outdim']
    nz = nets.setting['nz']

    # create the encoder architecture
    depth = int(opt.netE_type.split('-')[-1])
    has_masked_input = opt.masked or opt.vae_like
    assert(not (opt.masked and opt.vae_like)), "specify 1 of masked or vae_like"
    netE = biggan.load_BigGAN_encoder(domain=None, nz=nz,
                                                 outdim=out_shape,
                                                 use_RGBM=opt.masked,
                                                 use_VAE=opt.vae_like,
                                                 resnet_depth=depth,
                                                 ckpt_path=None)
    netE = netE.to(device).train()
    nets.encoder = netE
    print(netE)

    # losses + optimizers
    mse_loss = nn.MSELoss()
    # l1_loss = nn.L1Loss()
    perceptual_loss = losses.LPIPS_Loss(net='vgg', use_gpu=has_cuda)
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
        checkpoint = torch.load(opt.netE)
        netE.load_state_dict(checkpoint['state_dict'])
        optimizerE.load_state_dict(checkpoint['optimizer'])
        start_ep = checkpoint['epoch'] + 1
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']

    # uses 1600 samples per epoch, computes number of batches
    # based on batch size
    now = datetime.now()
    # one_class_only = True
    folder_to_save = 'training/biggan_'
    opt.orig_code = False
    if opt.orig_code:
        folder_to_save += 'orig_'
    else:
        folder_to_save += 'mse_perceptual_'
    if opt.masked:
        folder_to_save += 'masked_model_%d_' % mask_width
    if opt.mask_in_loss:
        folder_to_save += 'masked_mse_loss_mask_%d_' % mask_width
    if opt.one_class_only:
        class_number = 208
        folder_to_save += '_one_class_only_%d' % class_number
    folder_to_save += 'training_%s' % str(now.strftime("%d_%m_%Y_%H_%M_%S"))
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    #save opt arguments
    file_name = folder_to_save + r'/inf.txt'
    opt.file_name = file_name
    f = open(file_name,"w")
    now = datetime.now()
    f.write(str(now.strftime("%d/%m/%Y %H:%M:%S")))
    f.write("\n")
    f.write(folder_to_save)
    f.write("\n")
    f.write(str(opt))
    f.write("\n")
    f.close()


    epoch_batches = 1600 // batch_size
    total_loss = []
    mse_total_loss = []
    z_total_loss = []
    perceptual_total_loss = []
    z_norm_total = []
    encoded_norm_total = []
    for epoch, epoch_loader in enumerate(pbar(
        training_utils.epoch_grouper(train_loader, epoch_batches),
        total=(opt.niter-start_ep)), start_ep):
        # stopping condition
        if epoch > opt.niter:
            break
        # run a train epoch of epoch_batches batches
        for step, z_batch in enumerate(pbar(
            epoch_loader, total=epoch_batches), 1):

            z_batch = z_batch.to(device)
            # torch.manual_seed(0)
            z_batch = torch.normal(0, 1, size=[batch_size, nz]).to(device)
            if not opt.one_class_only:
                c = np.random.randint(low=0, high=999, size=(batch_size,))
            else:
                c = np.ones((batch_size,)) * class_number
            category = torch.Tensor([c]).long().cuda()
            c_shared = netG.shared(category).to(device)[0]
            fake_im = netG(z_batch,c_shared).detach()
            # for i in range(batch_size):
            #     plt.imshow(convert_image_np(fake_im[i,:,:,:].reshape(1,3,256,256)))
            #     plt.show()
            netE.zero_grad()
            if has_masked_input:
                # hints_fake, mask_fake = masking.mask_upsample(fake_im)
                mask_fake = rectangle_mask - 0.5
                hints_fake = fake_im * rectangle_mask
                encoded = netE(torch.cat([hints_fake, mask_fake], dim=1))
                if opt.masked:
                    encoded = encoded.reshape([batch_size, 120])
                    regenerated = netG(encoded,c_shared)
                elif opt.vae_like:
                    sample = torch.randn_like(encoded[:, nz:, :, :])
                    encoded_mean = encoded[:, nz:, :, :]
                    encoded_sigma = torch.exp(encoded[:, :nz, :, :])
                    reparam = encoded_mean + encoded_sigma * sample
                    regenerated = netG(reparam)
                    encoded = encoded_mean # just use mean in z loss
            else:
                # standard RGB encoding
                encoded = netE(fake_im)
                encoded = encoded.reshape([batch_size, 120])
                regenerated = netG(encoded,c_shared)

            # compute loss
            loss_z = training_utils.cor_square_error_loss(encoded, z_batch)
            loss_mse = mse_loss(regenerated, fake_im)
            loss_perceptual = perceptual_loss.forward(
                reshape(regenerated), reshape(fake_im)).mean()
            if opt.mask_in_loss:
                loss_mse = torch.mean((regenerated - fake_im) ** 2 * rectangle_mask)
            if opt.orig_code:

                loss = (opt.lambda_latent * loss_z + opt.lambda_mse * loss_mse
                        + opt.lambda_lpips * loss_perceptual)
            else:

                loss = (opt.lambda_mse * loss_mse  + opt.lambda_lpips * loss_perceptual)

            # optimize
            loss.backward()
            optimizerE.step()
            if opt.orig_code:
                z_total_loss.append(loss_z.detach().cpu().numpy())
            perceptual_total_loss.append(loss_perceptual.detach().cpu().numpy())
            mse_total_loss.append(loss_mse.detach().cpu().numpy())
            total_loss.append(loss.detach().cpu().numpy())
            z_norm_total.append(z_batch.norm(2,dim=1).mean().cpu())
            encoded_norm_total.append(encoded.norm(2,dim=1).mean().detach().cpu())
            # send losses to tensorboard
            if epoch % 5 ==0 and step % 25 == 0:
                total_batches = epoch * epoch_batches + step
                if opt.orig_code:
                    writer.add_scalar('%s/loss/train_z' % folder_to_save, loss_z, total_batches)
                    writer.add_scalar('%s/loss/train_lpips' % folder_to_save, loss_perceptual,
                                      total_batches)
                writer.add_scalar('%s/loss/train_mse' % folder_to_save, loss_mse, total_batches)
                writer.add_scalar('%s/loss/train_total' % folder_to_save, loss, total_batches)
                pbar.print("Epoch %d step %d mse %0.4f"
                           % (epoch, step, loss_mse.item()))
                x = np.linspace(0,len(total_loss)-1,len(total_loss))
                if opt.orig_code:
                    plt.plot(x,total_loss)
                    plt.plot(x,z_total_loss)
                    plt.plot(x,mse_total_loss)
                    plt.plot(x, perceptual_total_loss)
                    plt.legend(['total loss','z_loss','mse_loss','preceptual_loss'])
                else:
                    plt.plot(x, total_loss)
                    plt.plot(x,mse_total_loss)
                    plt.plot(x, perceptual_total_loss)
                    plt.legend(['total loss','mse_loss','preceptual_loss'])
                plt.title('losses')
                plt.savefig('%s/train_loss.jpg' % (folder_to_save))
                plt.close()

                plt.plot(x,z_norm_total)
                plt.plot(x,encoded_norm_total)
                plt.legend(['z_norm','encoded_norm'])
                plt.savefig('%s/z_norm.jpg' % (folder_to_save))
                plt.close()

                plt.figure(figsize=(20,10))
                if not opt.masked:
                    plt.subplot(2,4,1)
                    plt.imshow(convert_image_np(fake_im[0,:,:,:].reshape((1,3,resolution,resolution))))
                    plt.title('real image (input)')
                    plt.axis('off')
                    plt.subplot(2,4,5)
                    plt.imshow(convert_image_np(regenerated[0,:,:,:].detach().reshape((1,3,resolution,resolution))))
                    plt.title('generated image G(z)')
                    plt.axis('off')
                    plt.subplot(2,4,2)
                    plt.imshow(convert_image_np(fake_im[1,:,:,:].reshape((1,3,resolution,resolution))))
                    plt.axis('off')
                    plt.subplot(2,4,6)
                    plt.imshow(convert_image_np(regenerated[1,:,:,:].detach().reshape((1,3,resolution,resolution))))
                    plt.axis('off')
                    plt.subplot(2,4,3)
                    plt.imshow(convert_image_np(fake_im[2,:,:,:].reshape((1,3,resolution,resolution))))
                    plt.axis('off')
                    plt.subplot(2,4,7)
                    plt.imshow(convert_image_np(regenerated[2,:,:,:].detach().reshape((1,3,resolution,resolution))))
                    plt.axis('off')
                    plt.subplot(2, 4, 4)
                    plt.imshow(convert_image_np(fake_im[3, :, :, :].reshape((1, 3, resolution, resolution))))
                    plt.axis('off')
                    plt.subplot(2, 4, 8)
                    plt.imshow(convert_image_np(regenerated[3, :, :, :].detach().reshape((1, 3, resolution, resolution))))
                    plt.axis('off')
                else:
                    plt.subplot(3,4,1)
                    plt.imshow(convert_image_np(fake_im[0,:,:,:].reshape((1,3,resolution,resolution))))
                    plt.title('real image (input)')
                    plt.axis('off')
                    plt.subplot(3,4,5)
                    plt.imshow(convert_image_np(hints_fake[0,:,:,:].detach().reshape((1,3,resolution,resolution))))
                    plt.title('masked image')
                    plt.axis('off')
                    plt.subplot(3,4,9)
                    plt.imshow(convert_image_np(regenerated[0,:,:,:].detach().reshape((1,3,resolution,resolution))))
                    plt.title('generated image G(z)')
                    plt.axis('off')
                    plt.subplot(3,4,2)
                    plt.imshow(convert_image_np(fake_im[1,:,:,:].reshape((1,3,resolution,resolution))))
                    plt.axis('off')
                    plt.subplot(3,4,6)
                    plt.imshow(convert_image_np(hints_fake[1,:,:,:].detach().reshape((1,3,resolution,resolution))))
                    plt.axis('off')
                    plt.subplot(3,4,10)
                    plt.imshow(convert_image_np(regenerated[1,:,:,:].detach().reshape((1,3,resolution,resolution))))
                    plt.axis('off')
                    plt.subplot(3,4,3)
                    plt.imshow(convert_image_np(fake_im[2,:,:,:].reshape((1,3,resolution,resolution))))
                    plt.axis('off')
                    plt.subplot(3,4,7)
                    plt.imshow(convert_image_np(hints_fake[2,:,:,:].detach().reshape((1,3,resolution,resolution))))
                    plt.axis('off')
                    plt.subplot(3,4,11)
                    plt.imshow(convert_image_np(regenerated[2,:,:,:].detach().reshape((1,3,resolution,resolution))))
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
                plt.savefig('%s/train_images_epoch_%d_step_%d.jpg' % (folder_to_save,epoch,step))
                plt.close()
                #plt.show()
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

        # updated to run a small set of test zs 
        # rather than a single fixed batch
        netE.eval()
        test_metrics = {
            'loss_z': util.AverageMeter('loss_z'),
            'loss_mse': util.AverageMeter('loss_mse'),
            'loss_perceptual': util.AverageMeter('loss_perceptual'),
            'loss_total': util.AverageMeter('loss_total'),
        }
        for step, test_zs in enumerate(pbar(test_loader), 1):
            with torch.no_grad():
                test_zs = test_zs.to(device)
                test_zs = torch.normal(0, 1, size=[batch_size, nz]).to(device)
                c = np.random.randint(low=0, high=999, size=(batch_size,))
                category = torch.Tensor([c]).long().cuda()
                c_shared = netG.shared(category).to(device)[0]
                fake_im = netG(z_batch,c_shared)

                # fake_im = netG(test_zs)
                if has_masked_input:
                    hints_fake, mask_fake = masking.mask_upsample(fake_im)
                    encoded = netE(torch.cat([hints_fake, mask_fake], dim=1))
                    encoded = encoded.reshape([batch_size, 120])
                    if opt.masked:
                        regenerated = netG(encoded,c_shared)
                    elif opt.vae_like:
                        sample = torch.randn_like(encoded[:, nz:, :, :])
                        encoded_mean  = encoded[:, nz:, :, :]
                        encoded_sigma = torch.exp(encoded[:, :nz, :, :])
                        reparam = encoded_mean + encoded_sigma * sample
                        regenerated = netG(reparam)
                        encoded = encoded_mean # just use mean in z loss
                else:
                    encoded = netE(fake_im)
                    # regenerated = netG(encoded)
                    encoded = encoded.reshape([batch_size, 120])
                    regenerated = netG(encoded, c_shared)

                # compute loss
                loss_z = training_utils.cor_square_error_loss(encoded, test_zs)
                # loss_mse = mse_loss(regenerated, fake_im)
                loss_mse = torch.mean((regenerated - fake_im) ** 2 * rectangle_mask)
                loss_perceptual = (1/3) * perceptual_loss.forward(
                    reshape(regenerated), reshape(fake_im)).mean()

                # loss = (opt.lambda_latent * loss_z + opt.lambda_mse * loss_mse
                #         + opt.lambda_lpips * loss_perceptual)
                loss = loss_mse
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
        writer.add_scalar('loss/test_z', test_metrics['loss_z'].avg, epoch)
        writer.add_scalar('loss/test_mse', test_metrics['loss_mse'].avg, epoch)
        writer.add_scalar('loss/test_lpips', test_metrics['loss_perceptual'].avg, epoch)
        writer.add_scalar('loss/test_total', test_metrics['loss_total'].avg, epoch)
        writer.add_image('Test Image', grid, epoch)
        netE.train()

        # do checkpointing
        if epoch % 500 == 0 or epoch == opt.niter:
            training_utils.make_checkpoint(
                netE, optimizerE, epoch,
                test_metrics['loss_total'].avg.item(),
                '%s/netE_epoch_%d.pth' % (opt.outf, epoch))
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
                '%s/netE_epoch_best.pth' % (opt.outf))
            best_val_loss = test_metrics['loss_total'].avg.item()

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

    train(opt)
    elapsed = time.time() - t
    print(elapsed)