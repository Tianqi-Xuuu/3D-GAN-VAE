import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from models import Generator, Discriminator, ImageEncoder
from test import visualize_oblique_projection
from dataloader import DataLoader
from tqdm import trange


def train(args):
    # Set device to GPU or CPU
    device = args.device
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Initialize models: Generator, Discriminator, and Image Encoder
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    img_encoder = ImageEncoder().to(device)

    # Load pre-trained models if available
    if args.pretrained_generator:
        generator.load_state_dict(torch.load(args.pretrained_generator))
    if args.pretrained_discriminator:
        discriminator.load_state_dict(torch.load(args.pretrained_discriminator))
    if args.pretrained_img_encoder:
        img_encoder.load_state_dict(torch.load(args.pretrained_img_encoder))

    # Optimizers
    dis_optim = optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, betas=(args.beta, 0.999))
    gen_optim = optim.Adam(generator.parameters(), lr=args.generator_lr, betas=(args.beta, 0.999))
    img_optim = optim.Adam(img_encoder.parameters(), lr=args.generator_lr, betas=(args.beta, 0.999))

    # Loss functions
    bce_criterion = nn.BCELoss()  # Binary Cross Entropy for GAN losses
    mse_criterion = nn.MSELoss()  # Mean Squared Error for reconstruction loss

    # Load dataset using a PyTorch DataLoader
    dataset = DataLoader(args.dataset)
    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into train and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders for train and validation sets
    train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.tensorboard_path)

    dl, gl, il = [], [], []  # Lists to track discriminator, generator, and image encoder losses

    # Progress bar using tqdm
    progress_bar = trange(args.num_epochs, desc="Training Progress", unit="epoch")

    for epoch in progress_bar:
        for batch_idx, (real_models, real_imgs) in enumerate(train_loader):

            real_models = real_models.unsqueeze(1).to(device)  # (batch_size, 1, D, H, W)
            real_imgs = real_imgs.to(device)

            # ----------------------
            # Discriminator Training
            # ----------------------
            discriminator.train()
            dis_optim.zero_grad()

            z_random = torch.randn(real_models.shape[0], args.latent_dim, device=device)  # Random latent vector
            fake_models = generator(z_random)  # Fake models for GAN

            real_outputs = discriminator(real_models)  # Discriminator output for real models
            fake_outputs = discriminator(fake_models.detach())  # Discriminator output for fake models (detach)
            
            real_labels = torch.ones((real_models.shape[0], 1), device=device)
            fake_labels = torch.zeros((real_models.shape[0], 1), device=device)
            d_loss_real = bce_criterion(real_outputs, real_labels)
            d_loss_fake = bce_criterion(fake_outputs, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            # Calculate discriminator accuracy for logging
            real_accuracy = (real_outputs > 0.5).float().mean()
            fake_accuracy = (fake_outputs < 0.5).float().mean()
            accuracy = 0.5 * (real_accuracy + fake_accuracy)

            if accuracy.item() < 0.8:
                # Discriminator backward pass and step
                d_loss.backward()
                dis_optim.step()
            
            # ----------------------
            # Image Encoder (VAE) Training
            # ----------------------
            img_encoder.train()
            img_optim.zero_grad()

            for param in generator.parameters():
                param.requires_grad = False

            mean, logvar = img_encoder(real_imgs)  # Encoder outputs mean and log variance
            z_sampled = img_encoder.sample(mean, logvar)  # Sampling latent vector
            reconstructed_models = generator(z_sampled)  # Reconstructed 3D models from latent vector

            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / real_models.shape[0]
            # Reconstruction loss (between real and reconstructed models)
            mse_loss = mse_criterion(reconstructed_models, real_models)

            # Total VAE loss (KL + reconstruction)
            vae_loss = kl_loss * args.alpha[0] + mse_loss * args.alpha[1]
            vae_loss.backward()
            img_optim.step()

            for param in generator.parameters():
                param.requires_grad = True

            # ----------------------
            # Generator Training
            # ----------------------
            generator.train()
            gen_optim.zero_grad()

            # Generator tries to fool the discriminator (use non-detached fake_outputs)
            z_random = torch.randn(real_models.shape[0], args.latent_dim, device=device)
            fake_models = generator(z_random)
            fake_outputs_for_gan = discriminator(fake_models)  # Generator part for GAN
            g_loss_gan = bce_criterion(fake_outputs_for_gan, real_labels)  # GAN loss (fool discriminator)

            mean, logvar = img_encoder(real_imgs)  # Encoder outputs mean and log variance
            z_sampled = img_encoder.sample(mean, logvar).detach()  # Sampling latent vector
            reconstructed_models = generator(z_sampled)  # Reconstructed 3D models from latent vector
            mse_loss = mse_criterion(reconstructed_models, real_models)  # Reconstruction loss

            g_loss = g_loss_gan + mse_loss * args.alpha[1]

            g_loss.backward()
            gen_optim.step()

            # ----------------------
            # Logging
            # ----------------------
            dl.append(d_loss.item())
            gl.append(g_loss.item())
            il.append(vae_loss.item())

            # TensorBoard: log losses and accuracy
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
            writer.add_scalar("Loss/Generator", g_loss.item(), global_step)
            writer.add_scalar("Loss/ImageEncoder", vae_loss.item(), global_step)
            writer.add_scalar("Accuracy/Discriminator", accuracy.item(), global_step)

        # Calculate average losses
        avg_d_loss = sum(dl) / len(dl)
        avg_g_loss = sum(gl) / len(gl)
        avg_img_loss = sum(il) / len(il)

        # Update progress bar postfix with the average losses
        progress_bar.set_postfix({
            "d_loss": f"{avg_d_loss:.4f}",
            "g_loss": f"{avg_g_loss:.4f}",
            "img_loss": f"{avg_img_loss:.4f}"
        })


        if epoch % args.sample_epoch == 0 and epoch != 0:
            if not os.path.exists(args.sample_path):
                os.makedirs(args.sample_path)
            print('Validating...')

            sample_noise = torch.randn(args.batch_size, args.latent_dim, device=device)
            generated_volumes = generator(sample_noise).squeeze().cpu().detach().numpy()
            save_path = os.path.join(args.sample_path, 'models')
            save_path = os.path.join(save_path, f'epoch_{epoch}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(save_path + f'/epoch_{epoch}.npy', generated_volumes)
            for i in range(generated_volumes.shape[0]):
                visualize_oblique_projection(generated_volumes[i], save_path + f'/model_{i}.png')

            val_models, val_imgs = next(iter(val_loader))
            mean, logvar = img_encoder(val_imgs.to(device))
            z_sampled = img_encoder.sample(mean, logvar)
            reconstructed_models = generator(z_sampled).squeeze().cpu().detach().numpy()
            save_path = os.path.join(args.sample_path, 'reconstructed')
            save_path = os.path.join(save_path, f'epoch_{epoch}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            val_imgs = val_imgs.permute(0, 2, 3, 1).cpu().detach().numpy()
            for i in range(val_imgs.shape[0]):
                visualize_oblique_projection(reconstructed_models[i], save_path + f'/reconstructed_{i}.png', gt_img=val_imgs[i])


        if epoch % args.save_epoch == 0 and epoch != 0:
            if not os.path.exists(os.path.join(args.checkpoints_path, 'generator')):
                os.makedirs(os.path.join(args.checkpoints_path, 'generator'))
            if not os.path.exists(os.path.join(args.checkpoints_path, 'discriminator')):
                os.makedirs(os.path.join(args.checkpoints_path, 'discriminator'))
            if not os.path.exists(os.path.join(args.checkpoints_path, 'img_encoder')):
                os.makedirs(os.path.join(args.checkpoints_path, 'img_encoder'))
            torch.save(generator.state_dict(), os.path.join(args.checkpoints_path, f'generator/generator_epoch_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(args.checkpoints_path, f'discriminator/discriminator_epoch_{epoch}.pth'))
            torch.save(img_encoder.state_dict(), os.path.join(args.checkpoints_path, f'img_encoder/img_encoder_epoch_{epoch}.pth'))

    # Close the TensorBoard writer
    writer.close()


