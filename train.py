# Modified from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/ProGAN/train.py

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
    seed_everything
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config


def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )
    return loader, dataset


def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    writer,
):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        batch_size = real.shape[0]

        # Train the critic: maximize E[critic(real)] - E[critic(fake)]
        # turn to: minimize -E[critic(real)] - E[critic(fake)]
        # also include WGAN-GP losses
        noise = torch.randn(batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
        fake = gen(noise, alpha, step)
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))   # the main term
            + config.LAMBDA_GP * gp
            + (1e-3 * torch.mean(critic_real ** 2))
        )
        opt_critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        # Train the generator: maximize E[critric(fake)]
        # turn to: minimize -E[critic(fake)]
        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        if batch_idx % 100 == 0:
            with torch.no_grad():
                fake_imgs = gen(config.FIXED_NOISE, alpha, step)
                plot_to_tensorboard(writer, loss_critic.item(), loss_gen.item(), real.detach(), fake_imgs.detach(), tensorboard_step)
                tensorboard_step += 1

        # update alpha to fade in, after every epoch, the alpha adds 2 / num_epochs,
        # so after half of num_epochs of train_fn, alpha will reach 1
        alpha += batch_size / (config.PROGRESSIVE_EPOCHS[step] * 0.5 * len(dataset))
        alpha = min(alpha, 1)

        loop.set_postfix(loss_critic=loss_critic.item(), loss_gen=loss_gen.item())

    return tensorboard_step, alpha

def main():
    seed_everything()

    # models
    gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    critic = Discriminator(config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)

    # optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)

    writer = SummaryWriter('logs')

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE)

    gen.train()
    critic.train()

    tensorboard_step = 0
    # start at step that corresponds to img size that we set in config
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 0   # start with very low alpha
        img_size = (4 * 2 ** step)
        loader, dataset = get_loader(img_size)
        print('Current Image Size {}'.format(img_size))

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            tensorboard_step, alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
            )

            if config.SAVE_MODEL:   # save checkpoint at every epoch
                save_checkpoint(gen, opt_gen, img_size, epoch, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, img_size, epoch, filename=config.CHECKPOINT_CRITIC)

        generate_examples(gen, step, img_size)
        step += 1  # progress to the next img size

if __name__ == "__main__":
    main()