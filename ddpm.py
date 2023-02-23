import torch
from tqdm import tqdm
import logging


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    """ Diffusion """
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps                       # T = 1000
        self.beta_start = beta_start                         # b_start = 1e-4
        self.beta_end = beta_end                             # b_end = 0.02
        self.beta = self.prepare_noise_schedule().to(device) # b = [1e-4, ..., 0.02]
        self.alpha = 1. - self.beta                          # a = 1 - b
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)    # a_hat
        self.img_size = img_size                             # 256
        self.device = device                                 # 'cuda'

    def prepare_noise_schedule(self):
        """ Linearly increasing variance schedule """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """ 
        Forward Process
        ---------------
        Get q(x(t), x(t-1)) distribution at time step t
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]               # get (a[t]) ** 2 and add 4 channels
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None] # get (1-a[t])% ** 0.5 and add 4 channels
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise               # return (a_hat ** 0.5) * x + (1-a_hat ** 0.5) * e

    def sample_timesteps(self, n):
        """ return random timestep t_i """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        """ 
        Reverse Process 
        ---------------
        Sample from p(x(t-1), x(t)) distribution
        Gradually reconstruct image x from initialized x[0]=noise~N(0,1)
        Pass T times through model (t=T,...,1)
        """
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device) # begin with x=noise~N(0,1)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): # for t=T,...,1
                t = (torch.ones(n) * i).long().to(self.device) # timestep
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2    # to [0, 1]
        x = (x * 255).type(torch.uint8) # to [0, 255]
        return x
    
    def sample_conditional(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
    
class EMA:
    """ EMA """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())