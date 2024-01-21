import matplotlib.pyplot as plt
import torch
import math

def linear_schedule(beta_start = 1e-4, beta_end = 0.02, noise_steps = 1000):
    beta = torch.linspace(beta_start, beta_end, noise_steps)
    alpha = 1. - beta
    alphas_cumprod = torch.cumprod(alpha, dim=0)
    return alphas_cumprod




def cosine_schedule(noise_steps = 1000, s = 0.008):
    '''
    params:
    noise_steps: is the number of time steps.
    s: is the value of time step.
    '''
    '''
    return:
    beta_value: the beta value is worked the same as in ddpm
    '''
    steps = noise_steps + 1
    t = torch.linspace(0, noise_steps, steps, dtype = torch.float64) / noise_steps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod




def sigmoid_schedule(noise_steps = 1000, start = -3, end = 3, tau = 0.5, clamp_min = 1e-5):
    '''
    params:
    noise_steps: is the number of time steps.
    start: is the begin value of beta scheduler.
    end: is the final value of beta scheduler.
    tau: is the param of sigmoid function.
    clamp_min: is the smallest value which beta is gotten.
    '''
    '''
    return:
    beta_value: the beta value is worked the same as in ddpm
    '''
    steps = noise_steps + 1
    t = torch.linspace(0, noise_steps, steps, dtype = torch.float64) / noise_steps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod


alpha_cumprod_linear = linear_schedule()
alpha_cumprod_cosine = cosine_schedule()
alpha_cumprod_sigmoid = sigmoid_schedule()


plt.plot(alpha_cumprod_linear, label="linear", color="red")
plt.plot(alpha_cumprod_cosine, label="cosine", color="green")
plt.plot(alpha_cumprod_sigmoid, label="sigmoid", color="blue")
plt.legend(title="noise schedule")
plt.xlabel("t / T")
plt.ylabel("Cumulative Product Alpha")
plt.savefig("noise_schedule.png")