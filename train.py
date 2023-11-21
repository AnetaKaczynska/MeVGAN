import sys

from datetime import datetime
import argparse

from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from progan.models.utils.utils import load_progan
from models.frame_seed_generator import FrameSeedGenerator
from models.video_discriminator import VideoDiscriminator
from dataset.real_videos import RealVideos
from progan.visualization.visualizer import saveTensor


def train(fsg, progan, vdis, optimizer_D, optimizer_G, dataloader, start_epoch, epochs, name, log_writer):
    n_frames = 8
    time = torch.arange(n_frames).unsqueeze(1).cuda()

    for epoch in range(start_epoch, epochs):
        print(f'epoch: {epoch}')
        dis_loss, gen_loss = 0, 0
        total = 0
        dis_out_fakes, dis_out_reals = 0, 0
        acc_fakes, acc_reals = 0, 0
        for iter, real_video in tqdm(enumerate(dataloader)):
            real_video = real_video.cuda()

            # --------------- update discriminator ---------------

            optimizer_D.zero_grad()
            # ------ real input ------
            _, real_latent = progan.netD(real_video, getFeature=True)   # (N, 512)
            real_latent = real_latent.unsqueeze(0)                      # (1, N, 512)
            dis_real = vdis(real_latent)                                # (1, 1)
            label = torch.full([batch_size], 1., dtype=torch.float).cuda()
            errD_real = criterion(dis_real.squeeze(0), label)
            errD_real.backward()
            D_x = dis_real.mean().item()

            # ------ fake input ------
            noise = torch.randn([1, 2047]).tile(n_frames, 1).cuda()
            input = fsg(noise, time) 
            fake_video = progan.avgG(input)
            _, fake_latent = progan.netD(fake_video.detach(), getFeature=True)   # (N, 512)
            fake_latent = fake_latent.unsqueeze(0)                               # (1, N, 512)
            dis_fake = vdis(fake_latent)                                         # (1, 1)
            label.fill_(0.)
            errD_fake = criterion(dis_fake.squeeze(0), label)
            errD_fake.backward()
            D_G_z1 = dis_fake.mean().item()
            errD = errD_real + errD_fake
            optimizer_D.step()

            dis_out_fakes += D_G_z1
            dis_out_reals += D_x
            acc_fakes += 1 if round(D_G_z1) == 0 else 0
            acc_reals += 1 if round(D_x) == 1 else 0

            # --------------- update generator ---------------

            optimizer_G.zero_grad()
            _, fake_latent = progan.netD(fake_video, getFeature=True)   # (N, 512)
            dis_fake = vdis(fake_latent.unsqueeze(0))                   # (1, 512, N) -> (1, 1)
            label.fill_(1.)
            errG = criterion(dis_fake.squeeze(0), label)
            errG.backward()
            D_G_z2 = dis_fake.mean().item()
            optimizer_G.step()

            dis_loss += errD.item()
            gen_loss += errG.item()
            total += 1

            # --------------- log every 1000 iterations ---------------

            if iter % 1000 == 0:
                step = len(dataloader)*epoch+iter
                log_writer.add_scalar('Discriminator loss', dis_loss/total, step)
                log_writer.add_scalar('Generator loss', gen_loss/total, step)
                log_writer.add_scalar('Discriminator output/fakes', dis_out_fakes/total, step)
                log_writer.add_scalar('Discriminator output/reals', dis_out_reals/total, step)
                log_writer.add_scalar('Accuracy/fakes', acc_fakes/total, step)
                log_writer.add_scalar('Accuracy/reals', acc_reals/total, step)
                dis_loss, gen_loss = 0, 0
                total = 0
                dis_out_fakes, dis_out_reals = 0, 0
                acc_fakes, acc_reals = 0, 0

                fake_video = fake_video.detach().cpu()
                saveTensor(fake_video, (1024, 1024), f'fakes/{name}/video_{epoch}_{iter}.jpg')
                # real_video = real_video.detach().cpu()
                # saveTensor(real_video, (1024, 1024), f'reals/{name}/video_{epoch}_{iter}.jpg')
        
        torch.save(fsg.state_dict(), f'checkpoints/{name}/frame_seed_generator_epoch_{epoch}.pt')
        torch.save(vdis.state_dict(), f'checkpoints/{name}/video_discriminator_epoch_{epoch}.pt')
        torch.save(optimizer_G.state_dict(), f'checkpoints/{name}/optimizer_G_epoch_{epoch}.pt')
        torch.save(optimizer_D.state_dict(), f'checkpoints/{name}/optimizer_D_epoch_{epoch}.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='', help='Experiment name suffix')
    parser.add_argument('-u', '--unfreeze_pgan_disc', default=False, action='store_true', help='Unfreeze ProGAN Discriminator')
    parser.add_argument('-c', '--checkpoint_name', type=str, default=None, help='Checkpoint name to load model from (optional)')
    parser.add_argument('-e', '--checkpoint_epoch', type=int, default=None, help='Checkpoint epoch to load model from (optional)')
    parser.add_argument('-d', '--dataset_path', type=str, help='Path to dataset')
    args = parser.parse_args()

    now = datetime.now()
    date = now.strftime("%Y.%m.%d_%H.%M.%S")
    name = f'{date}_{args.name}'
    log_writer = SummaryWriter(f'runs/{name}')

    Path(f'fakes/{name}').mkdir(parents=True, exist_ok=True)
    # Path(f'reals/{name}').mkdir(parents=True, exist_ok=True)
    Path(f'checkpoints/{name}').mkdir(parents=True, exist_ok=True)

    # --------------- load all model components ---------------

    fsg = FrameSeedGenerator()
    vdis = VideoDiscriminator()
    if args.checkpoint_name and args.checkpoint_epoch:
        fsg.load_state_dict(torch.load(f'checkpoints/{args.checkpoint_name}/frame_seed_generator_epoch_{args.checkpoint_epoch}.pt'))
        vdis.load_state_dict(torch.load(f'checkpoints/{args.checkpoint_name}/video_discriminator_epoch_{args.checkpoint_epoch}.pt'))
    vdis.cuda()
    fsg.cuda()

    progan = load_progan('jelito3d_batchsize8', 'output_networks/jelito3d_batchsize8', freeze_pgan_disc=not(args.unfreeze_pgan_disc))

    # --------------- load optimizers ---------------

    criterion = nn.BCELoss()

    lr = 0.0002
    beta1 = 0.5
    optimizer_G = optim.Adam(fsg.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(vdis.parameters(), lr=lr, betas=(beta1, 0.999))

    if args.checkpoint_name and args.checkpoint_epoch:
        optimizer_G.load_state_dict(torch.load(f'checkpoints/{args.checkpoint_name}/optimizer_G_epoch_{args.checkpoint_epoch}.pt'))
        optimizer_D.load_state_dict(torch.load(f'checkpoints/{args.checkpoint_name}/optimizer_D_epoch_{args.checkpoint_epoch}.pt'))

    if args.unfreeze_pgan_disc:
        optimizer_D = optim.Adam(list(vdis.parameters()) + list(progan.netD.parameters()), lr=lr, betas=(beta1, 0.999))

    # --------------- load data ---------------

    real_videos = RealVideos(args.dataset_path)
    dataloader = DataLoader(real_videos, batch_size=None, shuffle=True)
    
    # --------------- train ---------------

    start_epoch = args.checkpoint_epoch + 1 if args.checkpoint_epoch else 0
    epochs = 100
    batch_size = 1

    fsg.train()
    vdis.train()

    if args.unfreeze_pgan_disc:
        progan.netD.train()
    else:
        progan.netD.eval()
    progan.avgG.eval()

    train(fsg, progan, vdis, optimizer_D, optimizer_G, dataloader, start_epoch, epochs, name, log_writer)
