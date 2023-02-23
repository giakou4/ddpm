import os
import argparse
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import save_images, get_data, setup_logging
from models import UNET
from ddpm import Diffusion


def parse_opt():
    """ Arguement Parser """
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_1"
    args.epochs = 500
    args.batch_size = 4
    args.image_size = 64
    args.dataset_path = r"../objects/UW-IS/LivingRoom/objects/bg"
    args.device = "cuda"
    args.lr = 3e-4
    args.ckpt = r"./ckpts/unconditional_ckpt.pt"
    return args


def train_one_epoch(diffusion, model, criterion, optimizer, loader, device, logger):
    """ One epoch training of DDPM """
    logging.info(f"Starting epoch {epoch}:")
    l = len(loader)
    pbar = tqdm(loader)
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        t = diffusion.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(images, t)
        predicted_noise = model(x_t, t)
        loss = criterion(noise, predicted_noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(MSE=loss.item()) # TQDM
        logger.add_scalar("MSE", loss.item(), global_step=epoch * l + batch_idx) # Tensorboard
        

if __name__ == '__main__':
    # Parse arguements
    args = parse_opt()
    setup_logging(args.run_name)
    device = args.device
    loader = get_data(args)
    
    # Define U-NET
    model = UNET()
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
    model = model.to(device)
    
    # Define DDPM
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    # Define Criterion and Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Define Tensorboard logger
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    
    # Training for epochs
    for epoch in range(args.epochs):
        train_one_epoch(diffusion, model, criterion, optimizer, loader, device, logger)
        if epoch % 10 == 0:
            sampled_images = diffusion.sample(model, n=4)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_{epoch}.pt"))
