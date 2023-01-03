# Modified from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/ProGAN/config.py

import torch

START_TRAIN_AT_IMG_SIZE = 4
DATASET = 'celeba_hq'
CHECKPOINT_GEN = 'last_generator_60.pth'
CHECKPOINT_CRITIC = 'last_critic_60.pth'
DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BETAS = (0.0, 0.99)
BATCH_SIZES = [256, 256, 256, 256, 128, 32, 16, 8, 4]
CHANNELS_IMG = 3
Z_DIM = 256
IN_CHANNELS = 256  # should be 512 in original paper
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [60] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4
LOGS = 'logs/epoch_60'
SAVE_EXAMPLES = 'saved_examples_60'