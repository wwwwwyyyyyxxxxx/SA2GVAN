#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchsummary import summary

from ss_baselines.common.utils import CategoricalNetWithMask
from ss_baselines.av_nav.models.rnn_state_encoder import RNNStateEncoder
from ss_baselines.av_wan.models.visual_cnn import VisualCNN, SA2VisualCNN
from ss_baselines.av_wan.models.map_cnn import MapCNN, SA2MapCNN
from ss_baselines.av_wan.models.audio_cnn import AudioCNN, SA2AudioCNN
from ss_baselines.common.utils import Flatten
from ss_baselines.av_nav.models.visual_cnn import conv_output_dim, layer_init
from torch.autograd import Function
# from zmq import device
import numpy as np
import torch.nn.functional as F

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):

    def __init__(self, net, dim_actions, masking=True):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNetWithMask(self.net.output_size, self.dim_actions, masking)
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
            self,
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=False,
    ):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, prev_actions, masks)
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, distribution

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, prev_actions, masks)
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class AudioNavBaselinePolicy(Policy):

    def __init__(self, observation_space, goal_sensor_uuid, masking, action_map_size, hidden_size=512, encode_rgb=False,
                 encode_depth=False):
        super().__init__(
            AudioNavBaselineNet(observation_space=observation_space, hidden_size=hidden_size,
                                goal_sensor_uuid=goal_sensor_uuid, encode_rgb=encode_rgb, encode_depth=encode_depth),
            # action_space.n,
            action_map_size ** 2,
            masking=masking)


class Net(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class AudioNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, encode_rgb, encode_depth):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._spectrogram = False
        self._gm = 'gm' in observation_space.spaces
        self._am = 'am' in observation_space.spaces

        self._spectrogram = 'spectrogram' == self.goal_sensor_uuid
        self.visual_encoder = VisualCNN(observation_space, hidden_size, encode_rgb, encode_depth)
        if self._spectrogram:
            self.audio_encoder = AudioCNN(observation_space, hidden_size)
        if self._gm:
            self.gm_encoder = MapCNN(observation_space, hidden_size, map_type='gm')
        if self._am:
            self.am_encoder = MapCNN(observation_space, hidden_size, map_type='am')

        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._hidden_size if self._spectrogram else 0) + \
                         (self._hidden_size if self._gm else 0) + \
                         (self._hidden_size if self._am else 0)
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        if 'rgb' in observation_space.spaces and encode_rgb:
            rgb_shape = observation_space.spaces['rgb'].shape
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        if 'depth' in observation_space.spaces and encode_depth:
            depth_shape = observation_space.spaces['depth'].shape
            summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        if 'spectrogram' in observation_space.spaces:
            audio_shape = observation_space.spaces['spectrogram'].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')
        if self._gm:
            gm_shape = observation_space.spaces['gm'].shape
            summary(self.gm_encoder.cnn, (gm_shape[2], gm_shape[0], gm_shape[1]), device='cpu')
        if self._am:
            am_shape = observation_space.spaces['am'].shape
            summary(self.am_encoder.cnn, (am_shape[2], am_shape[0], am_shape[1]), device='cpu')

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []

        if self._spectrogram:
            x.append(self.audio_encoder(observations))
        if self._gm:
            x.append(self.gm_encoder(observations))
        if self._am:
            x.append(self.am_encoder(observations))
        if not self.is_blind:
            x.append(self.visual_encoder(observations))

        x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        return x2, rnn_hidden_states1


class SA2AudioNavBaselineNet(Net):

    def __init__(self, observation_space, config, sound_num):
        super().__init__()
        self.goal_sensor_uuid = config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID
        self._hidden_size = config.RL.PPO.hidden_size
        self._spectrogram = False
        self._gm = 'gm' in observation_space.spaces
        self._am = 'am' in observation_space.spaces
        self.config = config.RL.PPO
        self.sound_num = sound_num

        self.is_visual_encode = config.RL.PPO['is_visual_encode']
        self.is_classify = config.RL.PPO['is_classify']
        self.is_visual_classify = config.RL.PPO['is_visual_classify']
        self.is_regression = config.RL.PPO['is_regression']
        self.reg_behind = hasattr(config.RL.PPO, "reg_behind") and config.RL.PPO.reg_behind
        self.double_cnn = hasattr(config.RL.PPO, "double_cnn") and config.RL.PPO.double_cnn
        self.ENCODE_RGB = config.ENCODE_RGB
        self.ENCODE_DEPTH = config.ENCODE_DEPTH

        self._spectrogram = 'spectrogram' == self.goal_sensor_uuid

        # start net definition
        if self.is_visual_encode:
            self.visual_encoder = SA2VisualCNN(observation_space, self._hidden_size, config.ENCODE_RGB,
                                               config.ENCODE_DEPTH, config.RL.PPO)
        else:
            self.ENCODE_RGB = False
            self.ENCODE_DEPTH = False

        self.audio_hidden_size = self._hidden_size
        self.gm_hidden_size = self._hidden_size

        if self._spectrogram:
            self.audio_encoder = SA2AudioCNN(observation_space, self.audio_hidden_size, config.RL.PPO)

        if self._gm:
            self.gm_encoder = SA2MapCNN(observation_space, self.gm_hidden_size, map_type='gm',
                                        config=config.RL.PPO)  # now dim(x_av = n)
        if self._am:
            self.am_encoder = SA2MapCNN(observation_space, self.gm_hidden_size, map_type='am', config=config.RL.PPO)
        self.rnn_input_size = 0
        if self._spectrogram:
            self.rnn_input_size += self.audio_hidden_size
        if self._gm:
            self.rnn_input_size += self.gm_hidden_size
        if self._am:
            self.rnn_input_size += self.gm_hidden_size
        if self.is_visual_encode and config.ENCODE_DEPTH:
            self.rnn_input_size += self._hidden_size
        self.state_encoder = RNNStateEncoder(self.rnn_input_size, self._hidden_size)

        if 'rgb' in observation_space.spaces and self.ENCODE_RGB:
            rgb_shape = observation_space.spaces['rgb'].shape
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        if 'depth' in observation_space.spaces and self.ENCODE_DEPTH:
            depth_shape = observation_space.spaces['depth'].shape
            summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        if 'spectrogram' in observation_space.spaces:
            audio_shape = observation_space.spaces['spectrogram'].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')
        if self._gm:
            gm_shape = observation_space.spaces['gm'].shape
            summary(self.gm_encoder.cnn, (gm_shape[2], gm_shape[0], gm_shape[1]), device='cpu')
        if self._am:
            am_shape = observation_space.spaces['am'].shape
            summary(self.am_encoder.cnn, (am_shape[2], am_shape[0], am_shape[1]), device='cpu')

        if hasattr(config.RL.PPO, 'classifier_behind'):
            self.classifier_behind = config.RL.PPO.classifier_behind
        else:
            self.classifier_behind = False
        if config.RL.PPO.is_classify:
            if config.RL.PPO.is_visual_classify or self.classifier_behind:
                self.classifier = Sound_Classifier2(2 * self._hidden_size, sound_num,
                                                    num_layers=config.RL.PPO.classifier_num_layers)
            else:
                self.classifier = Sound_Classifier2(self._hidden_size, sound_num,
                                                    num_layers=config.RL.PPO.classifier_num_layers)

        if config.RL.PPO.is_regression:
            self.regressor = RotRegresor(self._hidden_size, 3)
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        if self.is_visual_encode:
            return self.visual_encoder.is_blind
        else:
            return True

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, lambda_grad=1.0,):
        x = []
        x_gm = None
        x_am = None
        x_a = None
        x_v = None

        if self._spectrogram:
            x_a = self.audio_encoder(observations)
            x.append(x_a)
        if self._gm:
            x_gm = self.gm_encoder(observations)
            x.append(x_gm)

        if self._am:
            x_am = self.am_encoder(observations)
            x.append(x_am)

        if self.is_visual_encode:
            x_v = self.visual_encoder(observations)
            x.append(x_v)
        x_cm = None
        x_cm_gt = None

        x1 = torch.cat(x, dim=1)

        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        if self.is_classify:
            if self.is_visual_classify:
                x1 = grad_reverse(x1, lambda_grad)
                predicted_labels = self.classifier(x1)
            elif self.classifier_behind is False:
                x_a_g = grad_reverse(x_a, lambda_grad)
                predicted_labels = self.classifier(x_a_g)
            elif self.classifier_behind is True:
                x1_g = grad_reverse(x1, lambda_grad)
                predicted_labels = self.classifier(x1_g)
        else:
            predicted_labels = None

        if self.is_regression:
            if self.reg_behind:
                predicted_rot = self.regressor(x2)
            else:
                predicted_rot = self.regressor(x_a)
        else:
            predicted_rot = None
        return x2, rnn_hidden_states1, predicted_labels, predicted_rot,


class SA2AudioNavBaselinePolicy(Policy):

    def __init__(
            self,
            observation_space,
            sound_num=128,
            config=None):
        super().__init__(
            SA2AudioNavBaselineNet(
                observation_space=observation_space,
                config=config,
                sound_num=sound_num),
            (config.TASK_CONFIG.TASK.ACTION_MAP.MAP_SIZE) ** 2,
            masking=config.MASKING)
        self.config = config
        self.observation_space = observation_space

    def act(self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False, lambda_grad=1.0):
        features, rnn_hidden_states, predicted_labels, predicted_x_y, = self.net(observations, rnn_hidden_states,
                                                                                 prev_actions, masks,
                                                                                 lambda_grad=lambda_grad)
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, distribution, predicted_labels, predicted_x_y

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks, lambda_grad=1.0):
        features, _, _, _, _, _, _ = self.net(observations, rnn_hidden_states, prev_actions, masks, lambda_grad=lambda_grad)
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action, lambda_grad=1.0):
        features, rnn_hidden_states, predicted_labels, predicted_x_y, = self.net(observations, rnn_hidden_states,
                                                                                 prev_actions, masks,
                                                                                 lambda_grad=lambda_grad)
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, predicted_labels, predicted_x_y


class Sound_Classifier2(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers=2):
        super(Sound_Classifier2, self).__init__()
        self.net = []
        num_decrease_layers = int(np.ceil(np.log2(input_dim / output_dim)))
        for i in range(num_layers - num_decrease_layers):
            self.net.append(spectral_norm(nn.Linear(input_dim, input_dim)))
            self.net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        for i in range(min(num_decrease_layers, num_layers) - 1):
            self.net.append(spectral_norm(nn.Linear(input_dim, input_dim // 2)))
            self.net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            input_dim = input_dim // 2

        self.net.append(spectral_norm(nn.Linear(input_dim, output_dim)))
        self.net.append(nn.Softmax())
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class RotRegresor(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(RotRegresor, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(True), nn.Linear(1024, 1024), nn.ReLU(True),
                                 nn.Linear(1024, 512), nn.ReLU(True), nn.Linear(512, output_dim),
                                 nn.Tanh())

    def forward(self, x):
        return self.net(x)


class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.save_for_backward(x, lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        _, lambd = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * (-lambd)
        return grad_input, None


def grad_reverse(x, lambd):
    lambd = torch.autograd.Variable(torch.FloatTensor([lambd])).to(x.device)
    return GradReverse.apply(x, lambd)
