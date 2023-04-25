# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from distutils.command.config import config
import numpy as np
import torch
import torch.nn as nn

from ss_baselines.common.utils import Flatten
from ss_baselines.av_nav.models.visual_cnn import conv_output_dim, layer_init, add_normalize
from torch.nn.utils import spectral_norm


class AudioCNN(nn.Module):
    r"""A Simple 3-Conv CNN for processing audio spectrogram features

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """
    def __init__(self, observation_space, output_size, audiogoal_sensor):
        super(AudioCNN, self).__init__()
        # 2, 512, 'spectrogram'
        self._n_input_audio = observation_space.spaces[audiogoal_sensor].shape[2]
        self._audiogoal_sensor = audiogoal_sensor

        cnn_dims = np.array(observation_space.spaces[audiogoal_sensor].shape[:2], dtype=np.float32)

        if cnn_dims[0] < 30 or cnn_dims[1] < 30:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        for kernel_size, stride in zip(self._cnn_layers_kernel_size, self._cnn_layers_stride):
            cnn_dims = conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            #  nn.ReLU(True),
            Flatten(),
            nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size),
            nn.ReLU(True),
        )
        layer_init(self.cnn)

    def forward(self, observations):
        cnn_input = []
        audio_observations = observations[self._audiogoal_sensor]
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        audio_observations = audio_observations.permute(0, 3, 1, 2)
        cnn_input.append(audio_observations)

        cnn_input = torch.cat(cnn_input, dim=1)

        return self.cnn(cnn_input)


class SA2AudioCNN(nn.Module):
    r"""A Simple 3-Conv CNN for processing audio spectrogram features

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """
    def __init__(self, observation_space, output_size, audiogoal_sensor, ppo_cfg):
        super(SA2AudioCNN, self).__init__()
        # 2, 512, 'spectrogram'
        self._n_input_audio = observation_space.spaces[audiogoal_sensor].shape[2]
        self._audiogoal_sensor = audiogoal_sensor
        self.config = ppo_cfg
        self.normalize_config = ppo_cfg.normalize_config
        cnn_dims = np.array(observation_space.spaces[audiogoal_sensor].shape[:2], dtype=np.float32)

        if cnn_dims[0] < 30 or cnn_dims[1] < 30:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        for kernel_size, stride in zip(self._cnn_layers_kernel_size, self._cnn_layers_stride):
            cnn_dims = conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )
        if self.config.double_ear:
            self.cnn_left = nn.Sequential(
                add_normalize(nn.Conv2d(
                    in_channels=int(self._n_input_audio / 2),
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ), self.normalize_config),
                nn.ReLU(True),
                add_normalize(nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ), self.normalize_config),
                nn.ReLU(True),
                add_normalize(nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ), self.normalize_config),
                nn.ReLU(True),
                Flatten(),
                add_normalize(nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size), self.normalize_config),
                # nn.ReLU(True),
                # nn.BatchNorm1d(output_size, momentum=0.9)
            )
            self.cnn_right = deepcopy(self.cnn_left)
            self.cnn_mlp = add_normalize(nn.Linear(output_size * 2, output_size), self.normalize_config)
            layer_init(self.cnn_left)
            layer_init(self.cnn_right)
            #layer_init(self.cnn_mlp)
        else:
            self.cnn = nn.Sequential(
                add_normalize(nn.Conv2d(
                    in_channels=self._n_input_audio,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ), self.normalize_config),
                nn.ReLU(True),
                add_normalize(nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ), self.normalize_config),
                nn.ReLU(True),
                add_normalize(nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ), self.normalize_config),
                nn.ReLU(True),
                Flatten(),
                add_normalize(nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size), self.normalize_config),
                # nn.ReLU(True),
                # nn.BatchNorm1d(output_size, momentum=0.9)
            )
            layer_init(self.cnn)

    def forward(self, observations):
        cnn_input = []
        audio_observations = observations[self._audiogoal_sensor]
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        audio_observations = audio_observations.permute(0, 3, 1, 2)
        cnn_input.append(audio_observations)

        cnn_input = torch.cat(cnn_input, dim=1)
        if self.config.double_ear:
            cnn_input_left = cnn_input[:, 0, :, :].unsqueeze(dim=1)
            cnn_input_right = cnn_input[:, 1, :, :].unsqueeze(dim=1)
            f_l = self.cnn_left(cnn_input_left)
            f_r = self.cnn_right(cnn_input_right)
            f = torch.cat((f_l, f_r), dim=1)
            f = self.cnn_mlp(f)
            return f
        else:
            return self.cnn(cnn_input)


def layer_init_single(layer):

    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
        if layer.bias is not None:
            nn.init.constant_(layer.bias, val=0)
