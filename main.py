import warnings

from pyvirtualdisplay import Display

import copy
import torch
import random
import gym
import matplotlib
import functools
import itertools
import math

import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F

from collections import deque, namedtuple
from IPython.display import HTML
from base64 import b64encode

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from torch.distributions import Normal

from pytorch_lightning import LightningModule, Trainer

import brax
from brax import envs
from brax.envs import to_torch
from brax.io import html

""" set up GPU """
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()

v = torch.ones(1, device='cuda')


@torch.no_grad()
def create_video(env, episode_length, policy=None):
    qp_array = []
    state = env.reset()
    for i in range(episode_length):
        if policy:
            loc, scale = policy(state)
            sample = torch.normal(loc, scale)
            action = torch.tanh(sample)
        else:
            action = env.action_space.sample()
        state, _, _, _ = env.step(action)
        qp_array.append(env.unwrapped._state.qp)
    return HTML(html.render(env.unwrapped._env.sys, qp_array))


@torch.no_grad()
def test_agent(env, episode_length, policy, episodes=10):
    ep_returns = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            loc, scale = policy(state)
            sample = torch.normal(loc, scale)
            action = torch.tanh(sample)
            state, reward, done, info = env.step(action)
            ep_ret += reward.item()

        ep_returns.append(ep_ret)

    return sum(ep_returns) / episodes


# TODO: create the policy

class GradientPolicy(nn.Module):

    def __init__(self, in_features, out_dims, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, out_dims)
        self.fc_std = nn.Linear(hidden_size, out_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        loc = self.fc_mu(x)
        loc = torch.tanh(loc)
        scale = self.fc_std(x)
        scale = F.softplus(scale) + 0.001
        return loc, scale


# TODO: create the value network

class ValueNet(nn.Module):

    def __init__(self, in_features, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32).to(device)
        self.var = torch.ones(shape, dtype=torch.float32).to(device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation(gym.core.Wrapper):

    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape[-1])
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        obs = self.normalize(obs)
        return obs, rews, dones, infos

    def reset(self, **kwargs):
        return_info = kwargs.get("return_info", False)
        if return_info:
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
        obs = self.normalize(obs)
        if not return_info:
            return obs
        else:
            return obs, info

    def normalize(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + self.epsilon)


# TODO: create the environment
def create_env(env_name, num_envs=256, episode_length=1000):
    env = gym.make(env_name, batch_size=num_envs, episode_length=episode_length)
    env = to_torch.JaxToTorchWrapper(env, device=device)
    env = NormalizeObservation(env)
    return env


# TODO: create the dataset
class RLDataset(IterableDataset):
    def __init__(self, env, policy, value_net, samples_per_epoch,
                 gamma, lamb, repeats):
        self.samples_per_epoch = samples_per_epoch
        self.gamma = gamma
        self.lamb = lamb
        self.repeats = repeats
        self.env = env
        self.policy = policy
        self.value_net = value_net
        self.obs = self.env.reset()

    @torch.no_grad()
    def __iter__(self):
        transitions = []
        for step in range(self.samples_per_epoch):
            loc, scale = self.policy(self.obs)
            action = torch.normal(loc, scale)
            next_obs, reward, done, info = self.env.step(action)
            transitions.append((self.obs, loc, scale, action, reward, done, next_obs))
            self.obs = next_obs

        "create tensors for obs, loc, scale, action, reward, done, next_obs"
        transitions = map(torch.stack, zip(*transitions))
        obs_b, loc_b, scale_b, action_b, reward_b, done_b, next_obs_b = transitions
        # right now the reward_b and done_b are in the shape of
        # (samples_per_epoch, num_envs)
        # add an extra dim to make them compatible with the rest
        reward_b = reward_b.unsqueeze(dim=-1)
        done_b = done_b.unsqueeze(dim=-1)

        values_b = self.value_net(obs_b)
        next_value_b = self.value_net(obs_b)

        "create the batch of td errors"
        td_error_b = reward_b + (1 - done_b) * self.gamma * next_value_b - values_b

        "compute GAE"
        running_gae = torch.zeros((self.env.num_envs, 1), dtype=torch.float32, device=device)
        gae_b = torch.zeros_like(td_error_b)

        for row in range(self.samples_per_epoch - 1, -1, -1):
            running_gae = td_error_b[row] + (1 - done_b[row]) * self.gamma * self.lamb * running_gae
            gae_b[row] = running_gae

        target_b = gae_b + values_b  # r_t + gamma * next_state_values?

        "shuffle data and pass it to training"
        num_samples = self.samples_per_epoch * self.env.num_envs
        reshape_fn = lambda x: x.view(num_samples, -1)
        batch = [obs_b, loc_b, scale_b, action_b, reward_b, gae_b, target_b]

        obs_b, loc_b, scale_b, action_b, reward_b, gae_b, target_b = map(reshape_fn, batch)

        for repeat in range(self.repeats):
            idx = list(range(num_samples))
            random.shuffle(idx)

            for i in idx:
                yield obs_b[i], loc_b[i], scale_b[i], action_b[i], reward_b[i], gae_b[i], target_b[i]


# TODO: create PPO with Generalized Advantage Estimation
class PPO(LightningModule):
    # shrink samples_per_epoch from 10 to 8 for limited GPU memory
    def __init__(self, env_name, num_envs=2048, episode_length=1000,
                 batch_size=1024, hidden_size=192, samples_per_epoch=10,
                 policy_lr=1e-4, value_lr=1e-3, epoch_repeat=4, epsilon=0.3,
                 gamma=0.99, lamb=0.95, entropy_coef=0.1, optim=AdamW):

        super().__init__()

        self.env = create_env(env_name, num_envs=num_envs, episode_length=episode_length)
        test_env = gym.make(env_name, episode_length=episode_length)
        test_env = to_torch.JaxToTorchWrapper(test_env, device=device)
        self.test_env = NormalizeObservation(test_env)
        self.test_env.obs_rms = self.env.obs_rms

        obs_size = self.env.observation_space.shape[1]
        action_dims = self.env.action_space.shape[1]

        self.policy = GradientPolicy(obs_size, action_dims, hidden_size)
        self.value_net = ValueNet(obs_size, hidden_size)
        self.target_value_net = copy.deepcopy(self.value_net)

        self.dataset = RLDataset(self.env, self.policy, self.target_value_net,
                                 samples_per_epoch, gamma, lamb, epoch_repeat)

        self.save_hyperparameters()
        self.videos = []

    def configure_optimizers(self):
        value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)
        return value_opt, policy_opt

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset, batch_size=self.hparams.batch_size)

    def training_step(self, batch, batch_idx, optimizer_idx):
        obs_b, loc_b, scale_b, action_b, reward_b, gae_b, target_b = batch

        state_values = self.value_net(obs_b)

        if optimizer_idx == 0:
            loss = F.smooth_l1_loss(state_values, target_b)
            self.log("episode/Value Loss:", loss)
            return loss

        elif optimizer_idx == 1:

            new_loc, new_scale = self.policy(obs_b)
            dist = Normal(new_loc, new_scale)
            log_prob = dist.log_prob(action_b).sum(dim=-1, keepdim=True)

            prev_dist = Normal(loc_b, scale_b)
            prev_log_prob = prev_dist.log_prob(action_b).sum(dim=-1, keepdim=True)

            rho = torch.exp(log_prob - prev_log_prob)

            surrogate_1 = rho * gae_b
            surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * gae_b
            policy_loss = - torch.minimum(surrogate_1, surrogate_2)  # fix a negative sign here!!!
            # use the minus sign so that optimization
            # is in the right direction

            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            loss = policy_loss - self.hparams.entropy_coef * entropy

            self.log("episode/Policy Loss", policy_loss.mean())
            self.log("episode/Entropy", entropy.mean())
            self.log("episode/Reward", reward_b.mean())

            return loss.mean()

    def training_epoch_end(self, training_epoch_outputs):  # updating value network when epoch ends
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        if self.current_epoch % 10 == 0:
            average_return = test_agent(self.test_env, self.hparams.episode_length, self.policy, episodes=1)
            self.log("average/Average Return: ", average_return)

        if self.current_epoch % 50 == 0:
            video = create_video(self.test_env, self.hparams.episode_length, policy=self.policy)
            self.videos.append(video)


def main():
    warnings.filterwarnings('ignore')
    """set up virtual display"""
    Display(visible=False, size=(1400, 900)).start()

    entry_point = functools.partial(envs.create_gym_env, env_name='halfcheetah')
    gym.register('brax-halfcheetah-v0', entry_point=entry_point)

    # env = gym.make("brax-halfcheetah-v0", episode_length=1000)
    # env = to_torch.JaxToTorchWrapper(env, device=device)
    # create_video(env, 1000)
    env = create_env("brax-halfcheetah-v0", num_envs=10)
    # obs = env.reset()
    # print("Num envs:    ", obs.shape[0], "Obs dimension:   ", obs.shape[1])
    # print("Action space: ", env.action_space)
    # next_obs, reward, done, info = env.step(env.action_space.sample())
    # print("next obs: ", next_obs)
    # print("\n")
    # print("reward: ", reward)
    # print("\n")
    # print("done: ", done)
    # print("\n")
    # print("info: ", info)
    # print("\n")
    # print("info keys: ", info.keys())
    """
    Launch the training process! 
    """
    algo = PPO("brax-halfcheetah-v0")
    trainer = Trainer(gpus=num_gpus, max_epochs=500)

    trainer.fit(algo)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
