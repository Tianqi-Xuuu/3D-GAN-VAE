import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from dataloader import DataLoader
from models import Generator, Discriminator
from tqdm import trange

def train(args):
    # 设置设备为 GPU 或 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建生成器和判别器
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 优化器
    dis_optim = optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, betas=(args.beta, 0.999))
    gen_optim = optim.Adam(generator.parameters(), lr=args.generator_lr, betas=(args.beta, 0.999))

    # 损失函数
    criterion = nn.BCELoss()

    # 加载数据
    data_loader = DataLoader(args)
    X_train = np.array(data_loader.load_data()).astype(np.float32)
    X_train = torch.tensor(X_train).to(device)

    dl, gl = [], []

    # 进度条
    progress_bar = trange(args.num_epochs, desc="Training Progress", unit="epoch")

    for epoch in progress_bar:
        # 采样随机 batch
        idx = np.random.randint(len(X_train), size=args.batch_size)
        real_images = X_train[idx]

        # 添加第四个维度 (batch_size, 1, D, H, W)
        real_images = real_images.unsqueeze(1)
        # 生成随机噪声并生成假图像
        z = np.random.normal(0, 0.33, size=[args.batch_size, args.latent_dim]).astype(np.float32)
        z = torch.tensor(z).to(device)
        fake_images = generator(z)

        # 标签
        real_labels = torch.ones((args.batch_size, 1), device=device)
        fake_labels = torch.zeros((args.batch_size, 1), device=device)

        # ---------------------
        # 训练判别器
        # ---------------------
        discriminator.train()

        # 训练判别器使用真实图像
        dis_optim.zero_grad()
        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)

        # 训练判别器使用假图像
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)

        # 计算判别器总损失
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # 计算判别器的准确率
        real_accuracy = (real_outputs > 0.5).float().mean()
        fake_accuracy = (fake_outputs < 0.5).float().mean()
        accuracy = 0.5 * (real_accuracy + fake_accuracy)

        if accuracy < 0.8:
            d_loss.backward()
            dis_optim.step()

        # ---------------------
        # 训练生成器
        # ---------------------
        generator.train()

        gen_optim.zero_grad()
        z = np.random.normal(0, 0.33, size=[args.batch_size, args.latent_dim]).astype(np.float32)
        z = torch.tensor(z).to(device)

        fake_images = generator(z)

        # 生成器试图欺骗判别器，使判别器输出 1
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)

        g_loss.backward()
        gen_optim.step()

        # 保存损失
        dl.append(d_loss.item())
        gl.append(g_loss.item())

        avg_d_loss = round(sum(dl) / len(dl), 4)
        avg_g_loss = round(sum(gl) / len(gl), 4)

        # 更新进度条后缀以显示损失
        progress_bar.set_postfix({
            "d_loss_real": round(d_loss_real.item(), 4),
            "g_loss": round(g_loss.item(), 4),
            "avg_d_loss": avg_d_loss,
            "avg_g_loss": avg_g_loss
        })

        # ---------------------
        # 采样
        # ---------------------
        if epoch % args.sample_epoch == 0:
            if not os.path.exists(args.sample_path):
                os.makedirs(args.sample_path)
            print('Sampling...')
            sample_noise = np.random.normal(0, 0.33, size=[args.batch_size, args.latent_dim]).astype(np.float32)
            sample_noise = torch.tensor(sample_noise).to(device)
            generated_volumes = generator(sample_noise).cpu().detach().numpy()
            np.save(args.sample_path + f'/sample_epoch_{epoch+1}.npy', generated_volumes)

        # ---------------------
        # 保存权重
        # ---------------------
        if epoch % args.save_epoch == 0:
            if not os.path.exists(args.checkpoints_path):
                os.makedirs(args.checkpoints_path)
            torch.save(generator.state_dict(), os.path.join(args.checkpoints_path, f'generator_epoch_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(args.checkpoints_path, f'discriminator_epoch_{epoch+1}.pth'))
