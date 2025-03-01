import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader

manualSeed = 999
torch.manual_seed(manualSeed) # Creates image grid

# Hyperparameters
batch_size = 64        # Number of images per batch
image_size = 64        # Resizing image to 64x64
nc = 1                 # Number of channels (1 is for greyscale)
nz = 100               # Size of latent vector (input noise of the generator)
ngf = 64               # Generator feature map size
ndf = 64               # Discriminator feature map size
# ngf and ndf control the depth of the gen and disc
num_epochs = 25
lr = 0.0002 # Learning rate
beta1 = 0.5 # Momentum parameter of Adam optimizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset preparation
dataset = dset.ImageFolder(root="data/brain_mri",                                   # Class automatically labels images based on folder names (e.g., "Yes" and "No"
                           transform=transforms.Compose([
                               transforms.Grayscale(num_output_channels=nc),        # Ensuring the images have one channel
                               transforms.Resize(image_size),                       # Standardizing the image size
                               transforms.CenterCrop(image_size),                   
                               transforms.ToTensor(),                               # Converting the images to PyTorch tensors
                               transforms.Normalize((0.5,), (0.5,))                 # Scaling the pixel values to [-1,1] as we are using Tanh function in the last layer of the Generator
                           ]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Generator, we are performing upsampling, so the first image produced is a 4x4 and then each layer increased it to 64x64 this is to introduce precision
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)

# Discriminator, we are performing down sampling and then squash it down to between 0 to 1, to check whether it is real or fake
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input).view(-1)

# Initializing the weights of convolutional and batch normalization layers to small random values.
def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    # Creating Generator and Discriminator instances and applying the random weights
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Using BCE loss function and Adam optimizer to measure the loss and update the weights for generator and discriminator
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    real_label = 1.
    fake_label = 0.
    fixed_noise = torch.randn(64, nz, 1, 1, device=device) # Used to view the progression of the quality of images overtime.

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # Updating the Discriminator
            ############################
            netD.zero_grad()
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            labels = torch.full((b_size,), real_label, device=device)
            output = netD(real_images)
            errD_real = criterion(output, labels) # Calculating the loss from the real images
            errD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            labels.fill_(fake_label)
            output = netD(fake_images.detach())
            errD_fake = criterion(output, labels)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # Updating the Generator
            ############################
            netG.zero_grad()
            labels.fill_(real_label)  # Generator wants discriminator to output real
            output = netD(fake_images)
            errG = criterion(output, labels)
            errG.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")

        os.makedirs("output", exist_ok=True)
        vutils.save_image(fake_images.detach(), f"output/fake_samples_epoch_{epoch}.png", normalize=True)

    print("Training complete!")