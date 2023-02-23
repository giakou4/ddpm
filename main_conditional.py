import os
import argparse
import copy
import numpy as np
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import save_images, get_data, setup_logging, plot_images
from models import UNET_conditional
from ddpm import Diffusion, EMA


def parse_opt():
    """ Arguement Parser """
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional_1"
    args.epochs = 500
    args.batch_size = 4
    args.image_size = 64
    args.dataset_path = r"../objects/UW-IS/LivingRoom/objects/bg"
    args.device = "cuda"
    args.lr = 3e-4
    args.num_classes = 14
    args.ckpt = None
    return args


def train_one_epoch(diffusion, model, ema, ema_model, criterion, optimizer, loader, device, logger):
    """ One epoch training of DDPM """
    logging.info(f"Starting epoch {epoch}:")
    l = len(loader)
    pbar = tqdm(loader)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        t = diffusion.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(images, t)
        if np.random.random() < 0.1:
            labels = None
        predicted_noise = model(x_t, t, labels)
        loss = criterion(noise, predicted_noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.step_ema(ema_model, model)
        pbar.set_postfix(MSE=loss.item()) # TQDM
        logger.add_scalar("MSE", loss.item(), global_step=epoch * l + batch_idx) # Tensorboard
        

if __name__ == '__main__':
    # Parse arguements
    args = parse_opt()
    setup_logging(args.run_name)
    device = args.device
    loader = get_data(args)
    
    # Define U-NET
    model = UNET_conditional(num_classes=args.num_classes)
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
    model = model.to(device)
    
    # Define DDPM
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    # EMA
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    # Define Criterion and Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Define Tensorboard logger
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    
    # Training for epochs
    for epoch in range(args.epochs):
        train_one_epoch(diffusion, model, ema, ema_model, criterion, optimizer, loader, device, logger)
        # Plot occasionally
        if epoch % 30 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample_conditional(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample_conditional(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_{epoch}.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt_{epoch}.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim_{epoch}.pt"))
