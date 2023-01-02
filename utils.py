# Modified from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/ProGAN/utils.py

import torch
import random
import numpy as np
import os
import torchvision
import config
from torchvision.utils import save_image

def plot_to_tensorboard(
    writer, loss_critic, loss_gen, real, fake, tensorboard_step
):
    writer.add_scalar('Critic Loss', loss_critic, global_step=tensorboard_step)
    writer.add_scalar("Generator Loss", loss_gen, global_step=tensorboard_step)

    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image('Real Images', img_grid_real, global_step=tensorboard_step)
        writer.add_image('Fake Images', img_grid_fake, global_step=tensorboard_step)


def gradient_penalty(critic, real, fake, alpha, train_step, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, img_size, epoch, filename):
    print('=> Save Checkpoint for {}*{} images, at epoch {}'.format(img_size, img_size, epoch))
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'img_size': img_size,
        'epoch': epoch
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print('=> Load Checkpoint')
    checkpoint = torch.load(checkpoint_file, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Reset the original training parameters of the optimizer doesn't harm much, it 
    # might cause the model to need a few more rounds to catch up, but I think that
    # is ok
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    img_size, epoch = checkpoint['img_size'], checkpoint['epoch']
    print('Checkpoint for {}*{} images at epoch {} loaded'.format(img_size, img_size, epoch))

def seed_everything(seed=423):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_examples(gen, steps, img_size, n=100):
    '''
    Generate images for different img_sizes.
    '''
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, config.Z_DIM, 1, 1).to(config.DEVICE)
            img = gen(noise, alpha, steps)
            save_image(img*0.5+0.5, 'saved_examples/img_size_{}_{}.png'.format(img_size, i + 1))
    gen.train()