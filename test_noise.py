import torch
from torchvision.utils import save_image
from ddpm import Diffusion
from utils import get_data
import argparse


def parse_opt():
    """ Arguement Parser """
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 1
    args.image_size = 64
    args.dataset_path = r"../objects/UW-IS/LivingRoom/objects/bg"
    return args

if __name__ == '__main__':
    """ Main Function """
    args = parse_opt()
    loader = get_data(args)
    diff = Diffusion(device="cpu")
    x = next(iter(loader))[0]
    t = torch.Tensor([50, 100, 150, 200, 300, 600, 700, 999]).long()
    noised_image, _ = diff.noise_images(x, t)
    save_image(noised_image.add(1).mul(0.5), "noise.jpg")
