import torch
import torch.nn as nn

## i need to use the same seq and var name as the repo from where the dataset is taken so that the load_state_dict can identify the 
## layers and work well .

#nz = the dim of the random noise vector that we feed as input to the generator
#ngf = number of generator features ie the number of channels in the last layer of the generator other above layers are mutiples of this number
#ndf = same as ngf but for discriminator
#nc = the number of channels output by the generator => 3 for RGB


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()

        #pytorch refers each line in a nn.sequential in the index starting from 0
        self.main = nn.Sequential(
            # main.0-2 : nz -> ngf*8, spatial 1x1 -> 4x4
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),                
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            # main.3-5 : ngf*8 -> ngf*4, spatial 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            # main.6-8 : ngf*4 -> ngf*2, spatial 8x8 -> 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            # main.9-11 : ngf*2 -> ngf, spatial 16x16 -> 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            # main.12 : ngf -> nc, spatial stays 32x32 (1x1 conv = channel mix only).
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        # z: (batch, nz, 1, 1) -- ConvTranspose2d requires a 4D (N,C,H,W) input,
        # so the latent vector is reshaped to have a 1x1 "spatial" extent before
        # the first transposed conv expands it to 4x4.
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            # main.0-1 : nc -> ndf, spatial 32x32 -> 16x16
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # main.2-4 : ndf -> ndf*2, spatial 16x16 -> 8x8
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # main.5-7 : ndf*2 -> ndf*4, spatial 8x8 -> 4x4
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # main.8-10 : ndf*4 -> ndf*8, spatial 8x8 -> 4x4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # main.11 : ndf*8 -> 1, spatial 4x4 -> 1x1 (kernel=2, stride=2, pad=0)
            nn.Conv2d(ndf * 8, 1, kernel_size=2, stride=2, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img):
        out = self.main(img)         # shape (batch, 1, 1, 1)
        return out.view(-1)          # flatten to (batch,) to match BCELoss target shape


#the below was used for a check if the generator and discriminator are loading well
#it also puts in the pretrained weights to the model
if __name__ == "__main__":
    netG = Generator(nz=100, ngf=64, nc=3)
    state_dict_g = torch.load("./pretrained/netG_epoch_199.pth", map_location="cpu")
    netG.load_state_dict(state_dict_g)
    print("Generator loaded OK")

    netD = Discriminator(ndf=64, nc=3)
    state_dict_d = torch.load("./pretrained/netD_epoch_199.pth", map_location="cpu")
    netD.load_state_dict(state_dict_d)
    print("Discriminator loaded OK")

    z = torch.randn(2, 100, 1, 1)
    fake = netG(z)
    print("Generator output shape:", fake.shape)  # (2, 3, 32, 32)

    pred = netD(fake)
    print("Discriminator output shape:", pred.shape)  # (2,)
