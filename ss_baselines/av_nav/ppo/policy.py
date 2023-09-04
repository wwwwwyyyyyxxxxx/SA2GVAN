#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchsummary import summary

from ss_baselines.common.utils import CategoricalNet
from ss_baselines.av_nav.models.rnn_state_encoder import RNNStateEncoder
from ss_baselines.av_nav.models.visual_cnn import VisualCNN, SA2VisualCNN
from ss_baselines.av_nav.models.audio_cnn import AudioCNN, SA2AudioCNN
from torch.autograd import Function
from zmq import device

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):

    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(self.net.output_size, self.dim_actions)
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
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, prev_actions, masks)
        distribution = self.action_distribution(features)
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

    def __init__(self, observation_space, action_space, goal_sensor_uuid, hidden_size=512, extra_rgb=False):
        super().__init__(
            AudioNavBaselineNet(observation_space=observation_space, hidden_size=hidden_size, goal_sensor_uuid=goal_sensor_uuid, extra_rgb=extra_rgb),
            action_space.n,
        )


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

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, extra_rgb=False):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._audiogoal = False
        self._pointgoal = False
        self._n_pointgoal = 0

        if DUAL_GOAL_DELIMITER in self.goal_sensor_uuid:
            goal1_uuid, goal2_uuid = self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)
            self._audiogoal = self._pointgoal = True
            self._n_pointgoal = observation_space.spaces[goal1_uuid].shape[0]
        else:
            if 'pointgoal_with_gps_compass' == self.goal_sensor_uuid:
                self._pointgoal = True
                self._n_pointgoal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
            else:
                self._audiogoal = True

        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb)
        if self._audiogoal:
            if 'audiogoal' in self.goal_sensor_uuid:
                audiogoal_sensor = 'audiogoal'
            elif 'spectrogram' in self.goal_sensor_uuid:
                audiogoal_sensor = 'spectrogram'

            self.audio_encoder = AudioCNN(observation_space, hidden_size, audiogoal_sensor)

        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._n_pointgoal if self._pointgoal else 0) + (self._hidden_size if self._audiogoal else 0)
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        if 'rgb' in observation_space.spaces and not extra_rgb:
            rgb_shape = observation_space.spaces['rgb'].shape
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        if self._audiogoal:
            audio_shape = observation_space.spaces[audiogoal_sensor].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')

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

        if self._pointgoal:
            x.append(observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]])
        if self._audiogoal:
            x.append(self.audio_encoder(observations))
        if not self.is_blind:
            x.append(self.visual_encoder(observations))

        x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        return x2, rnn_hidden_states1


class SA2AudioNavBaselinePolicy(Policy):

    def __init__(self, observation_space, action_space, goal_sensor_uuid, hidden_size=512, extra_rgb=False, sound_num=128, config=None):
        super().__init__(
            SA2AudioNavBaselineNet(observation_space=observation_space,
                                   hidden_size=hidden_size,
                                   goal_sensor_uuid=goal_sensor_uuid,
                                   extra_rgb=extra_rgb,
                                   config=config,
                                   sound_num=sound_num),
            action_space.n,
        )
        self.config = config
        self.observation_space = observation_space

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        lambda_grad=1.0,
    ):
        features, rnn_hidden_states, predicted_labels, predicted_rot = self.net(observations,
                                                                                rnn_hidden_states,
                                                                                prev_actions,
                                                                                masks,
                                                                                lambda_grad=lambda_grad)
        distribution = self.action_distribution(features)
        value = self.critic(features)
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, distribution, predicted_labels, predicted_rot, None, None, None

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks, lambda_grad=1.0):
        features, _, _, _ = self.net(observations, rnn_hidden_states, prev_actions, masks, lambda_grad=lambda_grad)
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action, lambda_grad=1.0):
        features, rnn_hidden_states, predicted_labels, predicted_rot = self.net(observations,
                                                                                rnn_hidden_states,
                                                                                prev_actions,
                                                                                masks,
                                                                                lambda_grad=lambda_grad)
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, predicted_labels, predicted_rot


class SA2AudioNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, extra_rgb=False, config=None, sound_num=128):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._audiogoal = False
        self._pointgoal = False
        self._n_pointgoal = 0
        self.config = config
        self.sound_num = sound_num
        self.observation_space = observation_space
        if DUAL_GOAL_DELIMITER in self.goal_sensor_uuid:
            goal1_uuid, goal2_uuid = self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)
            self._audiogoal = self._pointgoal = True
            self._n_pointgoal = observation_space.spaces[goal1_uuid].shape[0]
        else:
            if 'pointgoal_with_gps_compass' == self.goal_sensor_uuid:
                self._pointgoal = True
                self._n_pointgoal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
            else:
                self._audiogoal = True

        self.visual_encoder = SA2VisualCNN(observation_space, hidden_size, extra_rgb, config)
        if self._audiogoal:
            if 'audiogoal' in self.goal_sensor_uuid:
                audiogoal_sensor = 'audiogoal'
            elif 'spectrogram' in self.goal_sensor_uuid:
                audiogoal_sensor = 'spectrogram'

            self.audio_encoder = SA2AudioCNN(observation_space, hidden_size, audiogoal_sensor, config)

        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._n_pointgoal if self._pointgoal else 0) + (self._hidden_size if self._audiogoal else 0)
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        if 'rgb' in observation_space.spaces and not extra_rgb:
            rgb_shape = observation_space.spaces['rgb'].shape
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        if (hasattr(self.config, 'classifier_behind')):
            self.classifier_behind = self.config.classifier_behind
        else:
            self.classifier_behind = False
        if self._audiogoal:
            audio_shape = observation_space.spaces[audiogoal_sensor].shape
        if self.config.is_classify:
            if self.config.is_visual_classify or self.classifier_behind:
                self.classifier = Sound_Classifier(2 * self._hidden_size, sound_num, num_layers=self.config.classifier_num_layers)  # new classifier!
            else:
                self.classifier = Sound_Classifier(self._hidden_size, sound_num, num_layers=self.config.classifier_num_layers)
        if self.config.is_regression:
            self.regressor = RotRegresor(self._hidden_size, 3)
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

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, lambda_grad=1.0):
        x = []

        if self._pointgoal:
            x.append(observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]])
        if self._audiogoal:
            x_a = self.audio_encoder(observations)
            x.append(x_a)
        if not self.is_blind:
            x_v = self.visual_encoder(observations)
            x.append(x_v)

        x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        if self.config.is_classify:
            if self.config.is_visual_classify:
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

        if self.config.is_regression:
            if self.config.reg_behind:
                predicted_rot = self.regressor(x2)
            else:
                predicted_rot = self.regressor(x_a)
        else:
            predicted_rot = None

        return x2, rnn_hidden_states1, predicted_labels, predicted_rot


class Sound_Classifier(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers=2):
        super(Sound_Classifier, self).__init__()
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
        self.net = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(True), nn.Linear(1024, 1024), nn.ReLU(True), nn.Linear(1024, 512), nn.ReLU(True),
                                 nn.Linear(512, output_dim), nn.Tanh())

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
