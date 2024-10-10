import os
import argparse
from train import train
from datetime import datetime


def main(args):
    if args.mode == 'train':
        print('Training...')
        train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--im_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=200)

    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--strides', type=int, default=2)

    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--generator_lr', type=float, default=0.0025)
    parser.add_argument('--discriminator_lr', type=float, default=1e-5)

    parser.add_argument('--alpha', type=tuple, default=(5, 1e-4))

    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--save_epoch', type=int, default=1000)
    parser.add_argument('--sample_epoch', type=int, default=500)
    parser.add_argument('--test_epoch', type=int, default=100)

    parser.add_argument('--dataset', type=str, default='chair')

    parser.add_argument('--pretrained_generator', type=str, default='/data/3Dgan/3D-GAN-keras/Saved/20241009-150012/checkpoints/generator_epoch_9001.pth')
    parser.add_argument('--pretrained_discriminator', type=str, default='/data/3Dgan/3D-GAN-keras/Saved/20241009-150012/checkpoints/discriminator_epoch_9001.pth')
    parser.add_argument('--pretrained_img_encoder', type=str, default='/data/3Dgan/3D-GAN-keras/Saved/20241009-150012/checkpoints/img_encoder_epoch_9001.pth')

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument('--checkpoints_path', type=str, default=f'./Saved/{time}/checkpoints/')
    parser.add_argument('--tensorboard_path', type=str, default=f'./Saved/{time}/tensorboard/')
    parser.add_argument('--sample_path', type=str, default=f'./Saved/{time}/sample')

    args = parser.parse_args()
    main(args)