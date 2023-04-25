#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import logging

import numpy as np
import habitat
import torch
from habitat import Config, Dataset
from habitat.utils.visualizations.utils import observations_to_image
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.av_wan.models.planner import Planner, SA2Planner
import matplotlib
import os
import time
from PIL import Image
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis


@baseline_registry.register_env(name="MapNavEnv")
class MapNavEnv(habitat.RLEnv):

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._config = config
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._previous_target_distance = None
        self._previous_action = None
        self._previous_observation = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS_DISTANCE
        super().__init__(self._core_env_config, dataset)

        self.planner = Planner(model_dir=self._config.MODEL_DIR, use_acoustic_map='ACOUSTIC_MAP' in config.TASK_CONFIG.TASK.SENSORS, masking=self._config.MASKING, task_config=config.TASK_CONFIG)
        torch.set_num_threads(1)
        self.num = 1

    def reset(self):
        self._previous_action = None

        observations = super().reset()
        self.planner.update_map_and_graph(observations)
        self.planner.add_maps_to_observation(observations)
        self._previous_observation = observations
        logging.debug(super().current_episode)

        self._previous_target_distance = self.habitat_env.current_episode.info["geodesic_distance"]
        return observations

    def step(self, *args, **kwargs):
        intermediate_goal = kwargs["action"]
        self._previous_action = intermediate_goal
        goal = self.planner.get_map_coordinates(intermediate_goal)

        stop = int(self._config.TASK_CONFIG.TASK.ACTION_MAP.MAP_SIZE**2 // 2) == intermediate_goal
        observation = self._previous_observation

        cumulative_reward = 0
        done = False
        reaching_waypoint = False
        cant_reach_waypoint = False

        if len(self._config.VIDEO_OPTION) > 0:
            rgb_frames = list()
            audios = list()
        for step_count in range(self._config.PREDICTION_INTERVAL):
            if step_count != 0 and not self.planner.check_navigability(goal):
                cant_reach_waypoint = True
                break
            action = self.planner.plan(observation, goal, stop=stop)
            observation, reward, done, info = super().step({"action": action})

            current_position = self._env.sim.get_agent_state().position.tolist()
            info['apl'] = []
            info['apl'].append(current_position)

            if len(self._config.VIDEO_OPTION) > 0:
                if "rgb" not in observation:
                    observation["rgb"] = np.zeros((self._config.DISPLAY_RESOLUTION, self._config.DISPLAY_RESOLUTION, 3))
                frame = observations_to_image(observation, info)
                rgb_frames.append(frame)
                audios.append(observation['audiogoal'])
            cumulative_reward += reward
            if done:
                self.planner.reset()
                observation = self.reset()
                break
            else:
                self.planner.update_map_and_graph(observation)
                # reaching intermediate goal
                x, y = self.planner.mapper.get_maps_and_agent_pose()[2:4]
                if (x - goal[0]) == (y - goal[1]) == 0:
                    reaching_waypoint = True
                    break

        if not done:
            self.planner.add_maps_to_observation(observation)
        self._previous_observation = observation
        info['reaching_waypoint'] = done or reaching_waypoint
        info['cant_reach_waypoint'] = cant_reach_waypoint

        info['waypoints_pos'] = goal

        if len(self._config.VIDEO_OPTION) > 0:
            assert len(rgb_frames) != 0
            info['rgb_frames'] = rgb_frames
            info['audios'] = audios

        return observation, cumulative_reward, done, info

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
        target_position = [goal.position for goal in self._env.current_episode.goals]
        distance = self._env.sim.geodesic_distance(current_position, target_position)
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

    def global_to_egocentric(self, pg):
        return self.planner.mapper.global_to_egocentric(*pg)

    def egocentric_to_global(self, pg):
        return self.planner.mapper.egocentric_to_global(*pg)


@baseline_registry.register_env(name="SA2MapNavEnv")
class SA2MapNavEnv(habitat.RLEnv):

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._config = config
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._previous_target_distance = None
        self._previous_action = None
        self._previous_observation = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS_DISTANCE
        super().__init__(self._core_env_config, dataset)

        self.planner = SA2Planner(model_dir=self._config.MODEL_DIR, use_acoustic_map='ACOUSTIC_MAP' in config.TASK_CONFIG.TASK.SENSORS, masking=self._config.MASKING, task_config=config.TASK_CONFIG, config=config)
        torch.set_num_threads(1)
        self.num = 1
        self.debug_count = 0
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.data_generation = hasattr(config, "DATA_GENERATION") and config.DATA_GENERATION
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

    def reset(self):
        self._previous_action = None

        observations = super().reset()
        self.planner.update_map_and_graph(observations)
        self.planner.add_maps_to_observation(observations)

        self._previous_observation = observations
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

        info['agent_config'] = self._env.sim.config
        info['start_index'] = self._env.sim._position_to_index(self._env.sim.config.AGENT_0.START_POSITION)
        info['goal_index'] = self._env.sim._position_to_index(self._env.sim.config.AGENT_0.GOAL_POSITION)
        info['oracle_action'] = self._env.sim.get_oracle_action()
        observations["info"] = info
        # return observations, info
        self.waypoint_id_max = -1
        self.current_waypoint_id = 0
        self.debug_count = 0
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        return observations

    def get_dis_and_rot(self):
        p1 = self._env._sim.graph.nodes[self._env._sim._receiver_position_index]['point']
        p2 = self._env._sim.graph.nodes[self._env._sim._source_position_index]['point']
        d = np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)
        if abs(p1[2] - p2[2]) < 1e-6:
            r = np.sign(p1[0] - p2[0]) * (np.pi / 2)
        else:
            r = np.arctan((p1[0] - p2[0]) / (p1[2] - p2[2]))
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

    def step(self, *args, **kwargs):
        if self.data_generation:
            observation, reward, done, info = super().step({"action": kwargs["action"]})
            info['oracle_action'] = self._env.sim.get_oracle_action()
            return observation, reward, done, info
        intermediate_goal = kwargs["action"]
        self._previous_action = intermediate_goal
        goal = self.planner.get_map_coordinates(intermediate_goal)

        stop = int(self._config.TASK_CONFIG.TASK.ACTION_MAP.MAP_SIZE**2 // 2) == intermediate_goal
        observation = self._previous_observation

        x_delta, y_delta, z_delta = self.get_x_y_z()
        current_sound_id = self._env.sim._current_sound_id
        sound_num = self._env.sim._sound_num

        cumulative_reward = 0
        done = False
        reaching_waypoint = False
        cant_reach_waypoint = False

        if len(self._config.VIDEO_OPTION) > 0:
            rgb_frames = list()
            audios = list()

        obs_list = []
        waypoint_id_list = []
        for step_count in range(self._config.PREDICTION_INTERVAL):
            if step_count != 0 and not self.planner.check_navigability(goal):
                cant_reach_waypoint = True
                break
            action = self.planner.plan(observation, goal, stop=stop)
            observation, reward, done, info = super().step({"action": action})

            current_position = self._env.sim.get_agent_state().position.tolist()
            info['apl'] = []
            info['apl'].append(current_position)

            if len(self._config.VIDEO_OPTION) > 0:
                if "rgb" not in observation:
                    observation["rgb"] = np.zeros((self._config.DISPLAY_RESOLUTION, self._config.DISPLAY_RESOLUTION, 3))
                frame = observations_to_image(observation, info)
                rgb_frames.append(frame)
                audios.append(observation['audiogoal'])
            cumulative_reward += reward
            if done:
                self.planner.reset()
                observation = self.reset()
                break
            else:
                self.planner.update_map_and_graph(observation)
                # reaching intermediate goal
                x, y = self.planner.mapper.get_maps_and_agent_pose()[2:4]
                if (x - goal[0]) == (y - goal[1]) == 0:
                    reaching_waypoint = True
                    break

        if not done:
            self.planner.add_maps_to_observation(observation)

        info['reaching_waypoint'] = done or reaching_waypoint
        info['cant_reach_waypoint'] = cant_reach_waypoint

        info['waypoints_pos'] = goal

        if len(self._config.VIDEO_OPTION) > 0:
            assert len(rgb_frames) != 0
            info['rgb_frames'] = rgb_frames
            info['audios'] = audios

        # Don't know if we need current info or the upcoming
        info['current_agent_position'] = self._env.sim.get_agent_state().position.tolist()
        info['current_sound_name'] = self._env.sim._current_sound
        info['current_sound_position'] = self._env.sim.config.AGENT_0.GOAL_POSITION
        info['current_agent_orientation'] = self._env.sim.get_orientation()
        info['agent_config'] = self._env.sim.config
        info['start_index'] = self._env.sim._position_to_index(self._env.sim.config.AGENT_0.START_POSITION)
        info['GOAL_index'] = self._env.sim._position_to_index(self._env.sim.config.AGENT_0.GOAL_POSITION)

        info['current_agent_dis_sound'], info['current_agent_rotation'] = self.get_dis_and_rot()
        info['x_delta'], info['y_delta'], info['z_delta'] = x_delta, y_delta, z_delta
        info['current_sound_id'] = current_sound_id
        info['sound_num'] = sound_num
        info['oracle_action'] = self._env.sim.get_oracle_action()

        self._previous_observation = observation
        self.debug_count += 1
        return observation, cumulative_reward, done, info

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

        target_position = [goal.position for goal in self._env.current_episode.goals]
        distance = self._env.sim.geodesic_distance(current_position, target_position)
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

    def global_to_egocentric(self, pg):
        return self.planner.mapper.global_to_egocentric(*pg)

    def egocentric_to_global(self, pg):
        return self.planner.mapper.egocentric_to_global(*pg)
