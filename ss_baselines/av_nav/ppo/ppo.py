#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itsdangerous import NoneAlgorithm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from habitat import Config, logger
from zmq import device
from torch.distributions import Categorical
import queue
import random
from torch.utils.data import DataLoader, dataset, TensorDataset

EPS_PPO = 1e-5
from .policy import grad_reverse
import numpy as np
from PIL import Image
import time


class PPO(nn.Module):

    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)
        self.actor_critic.net.train()
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                # 就是PPO cliped的Loss！
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ)
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (0.5 * torch.max(value_losses, value_losses_clipped).mean())
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                total_loss = (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

    def after_step(self):
        pass


class SA2PPO(nn.Module):

    def __init__(self, actor_critic, clip_param, ppo_epoch, num_mini_batch, value_loss_coef, entropy_coef, lr=None, regressor_lr=None, classifier_lr=None, eps=None, max_grad_norm=None, use_clipped_value_loss=True, use_normalized_advantage=True, config=None):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.config = config
        ordinary_params = []
        classifier_params = []
        regressor_params = []
        for pname, p in actor_critic.named_parameters():
            if pname.find("classifier") != -1:
                classifier_params += [p]
            elif pname.find("regressor") != -1:
                regressor_params += [p]
            else:
                ordinary_params += [p]

        self.optimizer = optim.Adam([{'params': ordinary_params}, {'params': regressor_params, 'lr': regressor_lr if regressor_lr is not None else lr}, {'params': classifier_params, 'lr': classifier_lr if classifier_lr is not None else lr}], lr=lr, eps=eps)
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage
        if self.config.use_buffer_train or self.config.use_buffer_more:
            self.buffer_spect = torch.zeros((self.config.buffer_maxlen, ) + actor_critic.observation_space.spaces["spectrogram"].shape, device="cpu")
            self.buffer_label = torch.zeros((self.config.buffer_maxlen, 1), dtype=torch.long, device="cpu")
            self.buffer_depth = torch.zeros((self.config.buffer_maxlen, ) + actor_critic.observation_space.spaces["depth"].shape, device="cpu")
            self.buffer_pointer = 0

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts, lambda_grad=1.0):
        self.actor_critic.net.train()

        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        classifier_loss_epoch = 0
        classifier_entropy_epoch = 0
        classifier_acc_epoch = 0
        regressor_loss_epoch = 0

        lambda_grad *= self.config.lambda_classifier
        lambda_regressor = self.config.lambda_regressor
        classifier_acc = 0
        classifier_num = 0
        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)

            for data_generator_idx, sample in enumerate(data_generator):
                (obs_batch, recurrent_hidden_states_batch, actions_batch, prev_actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, info, sound_id_batch, x_batch, y_batch, z_batch) = sample
                # Reshape to do in a single forward pass for all steps
                (values, action_log_probs, dist_entropy, _, predicted_labels, predicted_dir ) = \
                    self.actor_critic.evaluate_actions(
                        obs_batch,
                        recurrent_hidden_states_batch,
                        prev_actions_batch,
                        masks_batch,
                        actions_batch,
                        lambda_grad=lambda_grad,
                    )

                if self.config.use_buffer_train or self.config.use_buffer_more:
                    if self.buffer_pointer + len(obs_batch["spectrogram"]) > self.config.buffer_maxlen:
                        self.buffer_spect[self.buffer_pointer:] = obs_batch["spectrogram"][:self.config.buffer_maxlen - self.buffer_pointer].to("cpu")
                        self.buffer_depth[self.buffer_pointer:] = obs_batch["depth"][:self.config.buffer_maxlen - self.buffer_pointer].to("cpu")
                        self.buffer_label[self.buffer_pointer:] = sound_id_batch[:self.config.buffer_maxlen - self.buffer_pointer].to("cpu")
                        self.buffer_spect[:len(obs_batch["spectrogram"]) - self.config.buffer_maxlen + self.buffer_pointer] = obs_batch["spectrogram"][self.config.buffer_maxlen - self.buffer_pointer:].to("cpu")
                        self.buffer_depth[:len(obs_batch["spectrogram"]) - self.config.buffer_maxlen + self.buffer_pointer] = obs_batch["depth"][self.config.buffer_maxlen - self.buffer_pointer:].to("cpu")
                        self.buffer_label[:len(obs_batch["spectrogram"]) - self.config.buffer_maxlen + self.buffer_pointer] = sound_id_batch[self.config.buffer_maxlen - self.buffer_pointer:].to("cpu")
                    else:
                        self.buffer_spect[self.buffer_pointer:self.buffer_pointer + len(obs_batch["spectrogram"])] = obs_batch["spectrogram"].to("cpu")
                        self.buffer_depth[self.buffer_pointer:self.buffer_pointer + len(obs_batch["spectrogram"])] = obs_batch["depth"].to("cpu")
                        self.buffer_label[self.buffer_pointer:self.buffer_pointer + len(obs_batch["spectrogram"])] = sound_id_batch.to("cpu")
                    self.buffer_pointer = (self.buffer_pointer + len(obs_batch["spectrogram"])) % self.config.buffer_maxlen

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ)
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (0.5 * torch.max(value_losses, value_losses_clipped).mean())
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                total_loss = (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)

                z_info = None
                if predicted_dir is not None:
                    rot_label = x_y_to_sin_cos(x_batch, y_batch, z_batch)


                    regressor_loss = F.mse_loss(predicted_dir, rot_label)
                    logger.info(f"regressor_loss: {regressor_loss}")

                    regressor_loss_epoch += regressor_loss.item()
                    if e == self.ppo_epoch - 1:
                        logger.info(f"z_max: {torch.max(torch.abs(z_batch))}")
                        z_info = (torch.max(torch.abs(z_batch)), rot_label[:, 2], torch.max(torch.abs(rot_label[:, 2])))
                    total_loss += lambda_regressor * regressor_loss

                if predicted_labels is not None:
                    if self.config.use_buffer_train:
                        lambda_classifier = 1.0
                        index = torch.from_numpy(np.random.randint(0, self.buffer_label.shape[0], self.config.train_batch)).long()

                        sound_id_batch = self.buffer_label[index]
                        spect_batch = self.buffer_spect[index]
                        depth_batch = self.buffer_depth[index]

                        train_spectrograms_tensor = dict()
                        train_spectrograms_tensor["spectrogram"] = spect_batch.to(self.device)
                        train_label_tensor = torch.tensor(sound_id_batch).squeeze(dim=1).to(self.device)
                        feature_a = self.actor_critic.net.audio_encoder(train_spectrograms_tensor).squeeze(0)
                        feature_a_g = grad_reverse(feature_a, lambda_grad)
                        if self.config.classifier_behind is False:
                            predicted_labels = self.actor_critic.net.classifier(feature_a_g)
                        else:
                            train_depth_tensor = dict()
                            train_depth_tensor["depth"] = depth_batch.to(self.device)
                            feature_v = self.actor_critic.net.visual_encoder(train_depth_tensor).squeeze(0)
                            x1 = torch.cat([feature_v, feature_a], dim=1)  # dim = 2n
                            x1_g = grad_reverse(x1, lambda_grad)
                            predicted_labels = self.actor_critic.net.classifier(x1_g)
                        classifier_loss = F.cross_entropy(predicted_labels, train_label_tensor)
                        classifier_entropy = Categorical(probs=predicted_labels).entropy().mean()
                        classifier_loss -= self.config.lambda_classifier_entropy * classifier_entropy
                        predicted_labels_arg_max = torch.argmax(predicted_labels.detach(), dim=1)
                        classifier_acc += (predicted_labels_arg_max == train_label_tensor).sum()
                        classifier_num += self.config.train_batch
                        classifier_loss_epoch += classifier_loss.item()
                        classifier_entropy_epoch += classifier_entropy.item()
                        classifier_acc_epoch += classifier_acc.item()

                        if e == self.ppo_epoch - 1:
                            logger.info(f"classifier_loss: {classifier_loss}")
                            logger.info(f"classifier_entropy: {classifier_entropy}")
                            logger.info(f"classifier_acc: {classifier_acc/classifier_num}")
                            logger.info(f"classifier_dist: {predicted_labels.detach()[0]}")

                            # debug
                            if predicted_dir is not None:
                                logger.info(f"z_batch: {z_batch}")
                                logger.info(f"z_max: {torch.max(torch.abs(z_batch))}")
                                logger.info(f"z_sin_label: {rot_label[:,2]}")
                                logger.info(f"z_sin_max: {torch.max(torch.abs(rot_label[:,2]))}")

                        total_loss += lambda_classifier * classifier_loss

                    else:
                        lambda_classifier = 1.0

                        sound_id_batch = sound_id_batch.squeeze(-1)

                        # debug

                        classifier_loss = F.cross_entropy(predicted_labels, sound_id_batch)

                        classifier_entropy = Categorical(probs=predicted_labels).entropy().mean()
                        classifier_loss -= self.config.lambda_classifier_entropy * classifier_entropy

                        predicted_labels_arg_max = torch.argmax(predicted_labels.detach(), dim=1)
                        classifier_acc = (predicted_labels_arg_max == sound_id_batch).sum() / sound_id_batch.shape[0]

                        # debug
                        logger.info(f"classifier_loss: {classifier_loss}")
                        logger.info(f"classifier_entropy: {classifier_entropy}")
                        logger.info(f"classifier_acc: {classifier_acc}")

                        classifier_loss_epoch += classifier_loss.item()
                        classifier_entropy_epoch += classifier_entropy.item()
                        classifier_acc_epoch += classifier_acc.item()

                        if e == 0:
                            logger.info(f"classifier_dist: {predicted_labels.detach()[0]}")

                        total_loss += lambda_classifier * classifier_loss


                self.optimizer.zero_grad()

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                if hasattr(self.config, "use_buffer_more") and self.config.use_buffer_more:
                    self.more_train()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        classifier_loss_epoch /= num_updates
        classifier_entropy_epoch /= num_updates
        if classifier_num == 0:  # avoid divide by zero
            classifier_acc_epoch = 0
        else:
            classifier_acc_epoch /= (num_updates * classifier_num)  # debug fix classifier_acc display in Tensorboard
        regressor_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, classifier_loss_epoch, classifier_entropy_epoch, \
               classifier_acc_epoch, regressor_loss_epoch, z_info

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

    def after_step(self):
        pass

    def more_train(self):
        train_spectrograms_tensor = torch.stack(self.buffer_list_spect, dim=0).squeeze(dim=1)
        train_label_tensor = torch.tensor(self.buffer_list_label)
        deal_dataset = TensorDataset(train_spectrograms_tensor, train_label_tensor)
        train_loader = DataLoader(dataset=deal_dataset, batch_size=self.config.train_batch, shuffle=True)
        optimizer = optim.Adam(self.actor_critic.net.classifier.parameters(), lr=1e-4, weight_decay=1e-4)
        for epoch in range(self.config.num_epoch):
            train_loss = 0
            spects = dict()
            classifier_acc_num = 0
            classifier_num = 0
            for i, data in enumerate(train_loader):
                spects["spectrogram"], sound_id = data
                spects["spectrogram"] = spects["spectrogram"].to(self.device)
                sound_id = sound_id.to(self.device)
                feature = self.actor_critic.net.audio_encoder(spects).squeeze(0)
                predicted_labels = self.actor_critic.net.classifier(feature)
                classifier_loss = F.cross_entropy(predicted_labels, sound_id)
                optimizer.zero_grad()

                classifier_loss.backward()
                torch.nn.utils.clip_grad_value_(self.actor_critic.net.classifier.parameters(), 10)
                optimizer.step()
                train_loss += classifier_loss.item()
                predicted_labels_arg_max = torch.argmax(predicted_labels, dim=1)
                classifier_acc_num += (predicted_labels_arg_max == sound_id).sum()
                classifier_num += sound_id.shape[0]
            logger.info("more train epoch:{} acc:{}".format(epoch, classifier_acc_num / classifier_num))


def x_y_to_dir(x, y, n):
    eps = 1e-6
    z_rot = torch.where(torch.abs(x) < eps, torch.where(y > 0, np.pi / 2, -np.pi / 2), torch.arctan(y / x))
    z_rot = torch.where(x < -eps, z_rot + np.pi * ((n + 1) / n), z_rot + np.pi * (1 / n))
    z_class = torch.floor(z_rot / (np.pi / (n / 2)))
    z_class = z_class.type(torch.long).squeeze()
    dir_label = torch.where(z_class < 0, z_class + n, z_class)
    return dir_label


def x_y_to_rot(x, y):
    eps = 1e-6
    z_rot = torch.where(torch.abs(x) < eps, torch.where(y > 0, np.pi / 2, -np.pi / 2), torch.arctan(y / x))
    z_rot = torch.where(x < -eps, z_rot + np.pi, z_rot)
    z_rot = torch.where(z_rot > np.pi, z_rot - 2 * np.pi, z_rot)
    return z_rot / np.pi


def x_y_to_sin_cos(x, y, z):
    eps = 1e-5
    dist = torch.sqrt(x * x + y * y) + eps
    dist_z = torch.sqrt(x * x + y * y + z * z) + eps
    sin_batch = x / dist
    cos_batch = y / dist
    sinzbatch = z / dist_z
    ret = torch.cat([sin_batch, cos_batch, sinzbatch], dim=1)
    return ret
