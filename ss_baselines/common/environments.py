#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

import imp
import time
from typing import Optional, Type
import logging
import os
from skimage.measure import block_reduce
import librosa
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat import Config, Dataset
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.utils import euler_from_quaternion
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import random
import numpy as np
import pandas as pd
import pickle
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="AudioNavRLEnv")
class AudioNavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS_DISTANCE
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()
        logging.debug(super().current_episode)

        self._previous_target_distance = self.habitat_env.current_episode.info["geodesic_distance"]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = 0

        if self._rl_config.WITH_TIME_PENALTY:
            reward += self._rl_config.SLACK_REWARD

        if self._rl_config.WITH_DISTANCE_REWARD:
            current_target_distance = self._distance_target()
            # if current_target_distance < self._previous_target_distance:
            reward += (self._previous_target_distance - current_target_distance) * self._rl_config.DISTANCE_REWARD_SCALE
            self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
            logging.debug('Reaching goal!')

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_positions = [goal.position for goal in self._env.current_episode.goals]
        distance = self._env.sim.geodesic_distance(current_position, target_positions)
        return distance

    def _episode_success(self):
        if (self._env.task.is_stop_called
                # and self._distance_target() < self._success_distance
                and self._env.sim.reaching_goal):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id


@baseline_registry.register_env(name="SA2AudioNavRLEnv")
class SA2AudioNavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self.myconfig = config
        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS_DISTANCE
        self.num_step = 0
        self.audio_use = {}
        self.spects = dict()
        self._core_env_config.defrost()
        if hasattr(config, "DEPTH_NOISE_LEVEL"):
            self._core_env_config.SIMULATOR.DEPTH_NOISE_LEVEL = config.DEPTH_NOISE_LEVEL
            self._core_env_config.SIMULATOR.DEPTH_SENSOR.NOISE_MODEL = "RedwoodDepthNoiseModel"
        if hasattr(config, "AUDIO_NOISE_LEVEL"):
            self._core_env_config.SIMULATOR.AUDIO_NOISE_LEVEL = config.AUDIO_NOISE_LEVEL
        if hasattr(config, "DISTRACTOR_SOUND_DIR"):
            self._core_env_config.SIMULATOR.AUDIO.DISTRACTOR_SOUND_DIR = config.DISTRACTOR_SOUND_DIR
        if hasattr(config, "HAS_DISTRACTOR_SOUND"):
            self._core_env_config.SIMULATOR.HAS_DISTRACTOR_SOUND = config.HAS_DISTRACTOR_SOUND
        self._core_env_config.freeze()
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        self.num_step = 0

        observations = super().reset()
        logging.debug(super().current_episode)

        self._previous_target_distance = self.habitat_env.current_episode.info["geodesic_distance"]
        info = {}
        info['current_agent_position'] = self._env.sim.get_agent_state().position.tolist()
        info['current_sound_name'] = self._env.sim._current_sound
        info['current_sound_position'] = self._env.sim.config.AGENT_0.GOAL_POSITION
        info['current_agent_orientation'] = self._env.sim.get_orientation()
        info['current_agent_dis_sound'], info['current_agent_rotation'] = self.get_dis_and_rot()
        info['x_delta'], info['y_delta'], info['z_delta'] = self.get_x_y_z()
        info['sound_num'] = self._env.sim._sound_num
        info['current_sound_id'] = self._env.sim._current_sound_id
        observations["info"] = info
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        # STOP: 0, FORWARD: 1, LEFT: 2, RIGHT: 3
        current_agent_dis_sound, current_agent_rotation = self.get_dis_and_rot()
        x_delta, y_delta, z_delta = self.get_x_y_z()
        sound_num = self._env.sim._sound_num
        current_sound_id = self._env.sim._current_sound_id

        outputs = super().step(*args, **kwargs)
        outputs[3]['current_agent_position'] = self._env.sim.get_agent_state().position.tolist()
        outputs[3]['current_sound_name'] = self._env.sim._current_sound
        outputs[3]['current_sound_position'] = self._env.sim.config.AGENT_0.GOAL_POSITION
        outputs[3]['current_agent_orientation'] = self._env.sim.get_orientation()

        outputs[3]['current_agent_dis_sound'], outputs[3]['current_agent_rotation'] = current_agent_dis_sound, current_agent_rotation
        outputs[3]['x_delta'], outputs[3]['y_delta'], outputs[3]['z_delta'] = x_delta, y_delta, z_delta
        outputs[3]['sound_num'] = sound_num
        outputs[3]['current_sound_id'] = current_sound_id
        return outputs

    @staticmethod
    def compute_spectrogram(audio_data):
        def compute_stft(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
            stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
            return stft

        channel1_magnitude = np.log1p(compute_stft(audio_data[0]))
        channel2_magnitude = np.log1p(compute_stft(audio_data[1]))
        spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)

        return spectrogram

    def change_sound_and_get_spectrogram(self):
        if self.myconfig.feature_dir is not None:
            sound_name = os.listdir(self.myconfig.feature_dir)
        spectrograms = dict()

        for name in sound_name:
            self._env._sim._audiogoal_cache = dict()
            self._env._sim._spectrogram_cache = dict()
            self._env._sim._current_sound = name
            spectrogram = self._env._sim.get_current_spectrogram_observation(self.compute_spectrogram)
            spectrograms[name] = spectrogram
        df = pd.DataFrame([spectrograms])
        d, r = self.get_dis_and_rot()
        if not os.path.isdir("/data/AudioVisual/pre_train_data/{0}".format(self._env._sim.current_scene_name)):
            os.makedirs("/data/AudioVisual/pre_train_data/{0}".format(self._env._sim.current_scene_name))
        with open('/data/AudioVisual/pre_train_data/{0}/{1}_{2}.pkl'.format(self._env._sim.current_scene_name, d, r), 'wb') as f:
            pickle.dump(df, f)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = 0

        if self._rl_config.WITH_TIME_PENALTY:
            reward += self._rl_config.SLACK_REWARD

        if self._rl_config.WITH_DISTANCE_REWARD:
            current_target_distance = self._distance_target()
            # if current_target_distance < self._previous_target_distance:
            reward += (self._previous_target_distance - current_target_distance) * self._rl_config.DISTANCE_REWARD_SCALE
            self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
            logging.debug('Reaching goal!')

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_positions = [goal.position for goal in self._env.current_episode.goals]
        distance = self._env.sim.geodesic_distance(current_position, target_positions)
        return distance

    def _episode_success(self):
        if (self._env.task.is_stop_called
                # and self._distance_target() < self._success_distance
                and self._env.sim.reaching_goal):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id

    def get_dis_and_rot(self):
        p1 = self._env._sim.graph.nodes[self._env._sim._receiver_position_index]['point']
        p2 = self._env._sim.graph.nodes[self._env._sim._source_position_index]['point']
        d = np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)
        if abs(p1[2] - p2[2]) < 1e-6:
            r = np.sign(p1[0] - p2[0]) * (np.pi / 2)
        else:
            r = np.arctan2(p2[2] - p1[2], p2[0] - p1[0])
        r += self._env._sim.get_orientation() * (np.pi / 180)
        return d, r

    def get_x_y_z(self):
        p1 = self._env._sim.graph.nodes[self._env._sim._receiver_position_index]['point']
        p2 = self._env._sim.graph.nodes[self._env._sim._source_position_index]['point']
        orientation = self._env._sim.get_orientation()
        x = p1[0] - p2[0]
        y = p1[2] - p2[2]
        dx = x * np.cos(orientation) - y * np.sin(orientation)
        dy = x * np.sin(orientation) + y * np.cos(orientation)
        dz = p1[1] - p2[1]
        return dx, dy, dz
