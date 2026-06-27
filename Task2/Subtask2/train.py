import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.nn.utils.parametrizations import spectral_norm

from model import Generator, Discriminator

def get_dataloader(dataroot, batch_size, image_size=32, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),                         # [0, 1]
        #apply mean and std of 0.5 to all the three channels 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])    # -> [-1, 1] matching Generator's Tanh output range
    
                                     

    dataset = dset.CIFAR10(root=dataroot, train=True, download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
        # drop_last=True: a smaller final batch (left overs ie the total data % batch size) would compute batchnorm
        # statistics over fewer samples adding noise 
    )
    return dataloader

def apply_spectral_norm(module, use_spectral_norm):
    if not use_spectral_norm:
        return module
    for i, layer in enumerate(module.main):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            module.main[i] = spectral_norm(layer)
    return module


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outf, exist_ok=True)

    dataloader = get_dataloader(args.dataroot, args.batch_size)

    netG = Generator(nz=args.nz, ngf=args.ngf, nc=3).to(device)
    netD = Discriminator(ndf=args.ndf, nc=3).to(device)

    netG.load_state_dict(torch.load(args.netG_checkpoint, map_location=device))
    netD.load_state_dict(torch.load(args.netD_checkpoint, map_location=device))

    if args.spectral_norm:
        netG = apply_spectral_norm(netG, True)
        netD = apply_spectral_norm(netD, True)

    netG.to(device)
    netD.to(device)

    
    criterion = nn.BCELoss()  

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    real_label = 1.0
    fake_label = 0.0

    # z: (batch, nz, 1, 1) -- ConvTranspose2d requires a 4D (N,C,H,W) input,
    # so the latent vector is reshaped to have a 1x1 "spatial" extent before
    # the first transposed conv expands it to 4x4.
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    g_losses, d_losses = [], []

    for epoch in range(args.epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)

           
            # Update D: maximize log(D(real)) + log(1 - D(G(z)))
           
            netD.zero_grad()

            label_real = torch.full((b_size,), real_label, device=device)
            output_real = netD(real_imgs)
            loss_d_real = criterion(output_real, label_real)
            loss_d_real.backward()

            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake_imgs = netG(noise)
            label_fake = torch.full((b_size,), fake_label, device=device)
            # .detach(): we don't want this backward() call (which updates D)
            # to also compute/accumulate gradients into G's parameters. G's
            # update happens in its own separate step below.
            output_fake = netD(fake_imgs.detach())
            loss_d_fake = criterion(output_fake, label_fake)
            loss_d_fake.backward()

            loss_d = loss_d_real + loss_d_fake
            optimizerD.step()

            
            # Update G: maximize log(D(G(z)))  [non-saturating trick:equivalent to minimizing BCE(D(G(z)), real_label)]
           
            netG.zero_grad()

            # Re-run D on the SAME fake_imgs, but NOT detached this time 
            # gradients need to flow back through D into G's parameters.
            output_fake_for_g = netD(fake_imgs)
            label_for_g = torch.full((b_size,), real_label, device=device)
            loss_g = criterion(output_fake_for_g, label_for_g)
            loss_g.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f"[{epoch}/{args.epochs}][{i}/{len(dataloader)}] "
                      f"Loss_D: {loss_d.item():.4f}  Loss_G: {loss_g.item():.4f}")
                g_losses.append(loss_g.item())
                d_losses.append(loss_d.item())

        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake, f"{args.outf}/fake_samples_epoch_{epoch:03d}.png",
                           normalize=True, nrow=8)

        torch.save(netG.state_dict(), f"{args.outf}/netG_epoch_{epoch:03d}.pth")
        torch.save(netD.state_dict(), f"{args.outf}/netD_epoch_{epoch:03d}.pth")

    return g_losses, d_losses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="./data")
    parser.add_argument("--outf", default="./output")
    parser.add_argument("--netG_checkpoint", default="./pretrained/netG_epoch_199.pth")
    parser.add_argument("--netD_checkpoint", default="./pretrained/netD_epoch_199.pth")
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--spectral_norm", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
