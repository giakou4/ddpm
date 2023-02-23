# Denoise Diffusion Probabilistic Model

Denoise Diffusion Probabilistic Models (DDPMs) are generative models based on the idea of reversing a noising process. The idea is fairly simple: Given a dataset, make it more and more noisy with a deterministic process. Then, learn a model that can undo this process.

In this notebook, I re-implement the first and most fundamental paper to be familiar with when dealing with DDPMs: Denoising Diffusion Probabilistic Models (https://arxiv.org/pdf/2006.11239.pdf) by Ho et. al.
