# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
import random
from wheel_legged_gym.envs import LeggedRobot

from wheel_legged_gym.utils.terrain_ori import  Terrain
# from collections import deque
from wheel_legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift


class CowaFreeEnv(LeggedRobot):
    '''
    CowaFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05   # 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        # self.compute_observations()

    # TODO
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales

        noise_vec[:3] = 0
        noise_vec[3:7] = noise_scales.dof_pos  * self.obs_scales.dof_pos
        noise_vec[7:13] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[13:19] = 0 
        noise_vec[19:22] = noise_scales.ang_vel  * self.obs_scales.ang_vel
        noise_vec[22:25] = noise_scales.quat  * self.obs_scales.quat
        noise_vec[25:27] = 0
        return noise_vec


    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay

        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions

        return super().step(actions)

    # TODO
    def compute_observations(self):
        self.base_height_obs = self.base_height.unsqueeze(1)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.0,
            )
            * self.obs_scales.height_measurements
        )
        self.privileged_obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,  # 3
            (self.dof_pos[:, [0,1,3,4]] - self.default_dof_pos[:, [0,1,3,4]]) * self.obs_scales.dof_pos,  # 4 
            self.dof_vel * self.obs_scales.dof_vel,  # 6
            self.actions,  # 6
            # self.last_actions,  # 6
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            # self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.projected_gravity,
            self.L0, # 2
            self.dof_acc * self.obs_scales.dof_acc, # 6
            self.torques * self.obs_scales.torques, # 6self
            heights,   # 77
            (self.body_mass - self.body_mass.mean()).view(self.num_envs, 1),    # 1
            self.base_com, # 3
            # self.default_dof_pos
        ), dim=-1)

        # print(self.body_mass.shape, self.base_com.shape, self.default_dof_pos.shape)
        # 加入额外的privileged
        # if self.cfg.env.priv_observe_friction:
        #      # 1
        #     friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.domain_rand.friction_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                             (self.friction_coeffs[:, 0].unsqueeze(1)
        #                                             - friction_coeffs_shift) * friction_coeffs_scale),dim=1)
        
        # if self.cfg.env.priv_observe_restitution:
        #     # 1
        #     restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.domain_rand.restitution_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.restitutions[:, 0].unsqueeze(1) 
        #                                          - restitutions_shift) * restitutions_scale),dim=1)

        # if self.cfg.env.priv_observe_base_mass:
        #     # 1
        #     payloads_scale, payloads_shift = get_scale_shift(self.cfg.domain_rand.added_mass_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),dim=1)

        # if self.cfg.env.priv_observe_com_displacement:
        #     # 3
        #     com_displacements_scale, com_displacements_shift = get_scale_shift(
        #         self.cfg.domain_rand.com_displacement_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.com_displacements - com_displacements_shift) * com_displacements_scale), dim=1)
        # if self.cfg.env.priv_observe_motor_strength:
        #     # 6
        #     motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.domain_rand.motor_strength_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale), dim=1)
 
        # if self.cfg.env.priv_observe_motor_offset:
        #     # 6
        #     motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.domain_rand.motor_offset_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.motor_offsets - motor_offset_shift) * motor_offset_scale), dim=1) 
        # if self.cfg.env.priv_observe_gravity:
        #     # 3
        #     gravity_scale, gravity_shift = get_scale_shift(self.cfg.domain_rand.gravity_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.gravities - gravity_shift) / gravity_scale), dim=1)

        # 如果速度预测，加变量
        # if self.cfg.env.num_est_prob is not None:
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                            self.base_lin_vel * self.obs_scales.lin_vel, 
                                            self.base_height_obs * self.obs_scales.height_measurements), dim=1)
        # agibot
        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_lag_timestep > self.last_dof_lag_timestep + 1
                self.dof_lag_timestep[cond] = self.last_dof_lag_timestep[cond] + 1
                self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_lag_buffer[torch.arange(self.num_envs), :, self.dof_lag_timestep.long()]
            self.lagged_dof_vel = self.dof_vel_lag_buffer[torch.arange(self.num_envs), :, self.dof_lag_timestep.long()]
        else:
            self.lagged_dof_pos = self.dof_pos
            self.lagged_dof_vel = self.dof_vel

        if self.cfg.domain_rand.add_imu_lag:
            if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.long()]
            self.lagged_base_ang_vel = self.lagged_imu[:,:3].clone()
            self.lagged_base_euler_xyz = self.lagged_imu[:,-3:].clone()
        # no imu lag
        else:              
            self.lagged_base_ang_vel = self.base_ang_vel[:,:3]
            self.lagged_base_euler_xyz = self.base_euler_xyz[:,-3:]
        q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.lagged_dof_vel * self.obs_scales.dof_vel

        obs_buf = torch.cat((
            self.commands[:, :3]  * self.commands_scale,   # 3 
            q[:, [0,1,3,4]],
            dq,  # 6D
            self.actions,   # 6D
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,  # 3
            # self.lagged_base_euler_xyz * self.obs_scales.quat,  # 3
            self.projected_gravity,#3
            # self.L0, # 2(左右腿长)
        ), dim=-1)
        obs_now = obs_buf.clone()

        if self.cfg.domain_rand.randomize_obs_delay:    # 实际是延迟2ms,这里是obs延迟一帧，延迟了10ms（扩大了）
            obs_now = self.hist_obs.popleft().to(self.device)
            self.hist_obs.append(obs_buf)

        if self.add_noise:  
            # print('obs buf =', obs_buf.shape)
            # print('self.noise_scale_vec =', self.noise_scale_vec.shape)
            add_noise = torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            obs_now += add_noise

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

# ================================================ Rewards Humanoid Gym ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        hip_roll_yaw_pitch_indices = [0,1,2,6,7,8]
        diff[:,hip_roll_yaw_pitch_indices] *= 3
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_leg_roll_joint_pos_outside(self):
        """
        防止leg roll外阔
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        r = 0
        diff_left_leg_roll = joint_pos[:, 0] - pos_target[:, 0]
        diff_right_leg_roll = joint_pos[:, 6] - pos_target[:, 6]
        mask_diff_left_leg_roll = diff_left_leg_roll <= 0
        mask_diff_right_leg_roll = diff_right_leg_roll >= 0
        diff_left_leg_roll[mask_diff_left_leg_roll] = 0
        diff_right_leg_roll[mask_diff_right_leg_roll] = 0
        
        temp_left = -1 * torch.exp(-2 * torch.norm(diff_left_leg_roll, dim=-1)) + 0.2 * torch.norm(diff_left_leg_roll, dim=-1).clamp(0, 0.5) + 0.6
        r += temp_left
        temp_right = -1 * torch.exp(-2 * torch.norm(diff_right_leg_roll, dim=-1)) + 0.2 * torch.norm(diff_right_leg_roll, dim=-1).clamp(0, 0.5) + 0.6
        r += temp_right

        return r

    def _reward_leg_roll_joint_pos_inside(self):
        """
        防止leg roll内收
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        r = 0
        diff_left_leg_roll = joint_pos[:, 0] - pos_target[:, 0]
        diff_right_leg_roll = joint_pos[:, 6] - pos_target[:, 6]
        mask_diff_left_leg_roll = diff_left_leg_roll >= 0
        mask_diff_right_leg_roll = diff_right_leg_roll <= 0
        diff_left_leg_roll[mask_diff_left_leg_roll] = 0
        diff_right_leg_roll[mask_diff_right_leg_roll] = 0
        
        temp_left = -1 * torch.exp(-2 * torch.norm(diff_left_leg_roll, dim=-1)) + 0.2 * torch.norm(diff_left_leg_roll, dim=-1).clamp(0, 0.5) + 0.6
        r += temp_left
        temp_right = -1 * torch.exp(-2 * torch.norm(diff_right_leg_roll, dim=-1)) + 0.2 * torch.norm(diff_right_leg_roll, dim=-1).clamp(0, 0.5) + 0.6
        r += temp_right

        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        self.foot_dist = foot_dist.unsqueeze(1)
        fd = self.cfg.rewards.min_feet_dist
        max_fd = self.cfg.rewards.max_feet_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_fd, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        knee_pos = self.rigid_state[:, self.knee_indices, :2]
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
        self.knee_dist = knee_dist.unsqueeze(1)
        fd = self.cfg.rewards.min_knee_dist
        max_df = self.cfg.rewards.max_knee_dist # / 1.5
        d_min = torch.clamp(knee_dist - fd, -0.5, 0.)
        d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2 # 只要是在[fd, max_df]，奖励就拿满

    # def _reward_orientation(self):
    #     """
    #     Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
    #     from the desired base orientation using the base euler angles and the projected gravity vector.
    #     """
    #     quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
    #     orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
    #     return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        # print(self.contact_forces[0, self.feet_indices, :])
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_dof_pos
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6: 8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.04, 0, 50)
        return torch.exp(-yaw_roll * 100)


    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error
    
    # only consider vel_x
    def _reward_tracking_lin_vel_x(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_x_error = torch.square(self.commands[:,0] - self.base_lin_vel[:,0])
        if self.cfg.rewards.tracking_vel_hard:
            return torch.exp(-torch.abs(self.commands[:,0] - self.base_lin_vel[:,0]) * 20)
        if self.cfg.rewards.tracking_vel_enhance:
            return 0.2 * torch.exp(-lin_vel_x_error * self.cfg.rewards.tracking_sigma_vel_x) + 0.8 * torch.exp(-lin_vel_x_error * 6 * self.cfg.rewards.tracking_sigma_vel_x)
        else:
            return torch.exp(-lin_vel_x_error * self.cfg.rewards.tracking_sigma_vel_x)
        # return torch.exp(-lin_vel_x_error * self.cfg.rewards.tracking_sigma) + torch.exp(-lin_vel_x_error * 10 * self.cfg.rewards.tracking_sigma) - 1
      
    # def _reward_tracking_ang_vel(self):
    #     """
    #     Tracks angular velocity commands for yaw rotation.
    #     Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
    #     """   
        
    #     ang_vel_error = torch.square(
    #         self.commands[:, 1] - self.base_ang_vel[:, 2])
    #     # print('ang_vel :', self.base_ang_vel[:, 2])
    #     if self.cfg.rewards.tracking_vel_enhance:
    #         return 0.2 *torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma_vel_y) + 0.8 *torch.exp(-ang_vel_error * 6 * self.cfg.rewards.tracking_sigma_vel_y)
    #     else:
    #         return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma_vel_y)

    def _reward_feet_height_smoothness(self):
        """
        计算feet height的平滑程度
        """
        r = torch.sum(torch.square(self.feet_height - self.last_feet_z), dim=1)
        return r

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.8 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)        
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        hip_knee_indices = [0,1,3,4]
        wheel_indices = [2,5]
        hip_knee_term_1 = torch.sum(torch.square(
            self.last_actions[:,hip_knee_indices]- self.actions[:,hip_knee_indices]), dim=1)
        hip_knee_term_2 = torch.sum(torch.square(
            self.actions[:,hip_knee_indices] + self.last_last_actions[:,hip_knee_indices] - 2 * self.last_actions[:,hip_knee_indices]), dim=1)
        hip_knee_term_3 = 0.2 * torch.sum(torch.abs(self.actions[:,hip_knee_indices]), dim=1)
        hip_knee_term = hip_knee_term_1 + hip_knee_term_2 + hip_knee_term_3
        
        wheel_term_1 = torch.sum(torch.square(
            self.last_actions[:,wheel_indices]- self.actions[:,wheel_indices]), dim=1)
        wheel_term_2 = torch.sum(torch.square(
            self.actions[:,wheel_indices] + self.last_last_actions[:,wheel_indices] - 2 * self.last_actions[:,wheel_indices]), dim=1)
        wheel_term_3 = 0.05 * torch.sum(torch.abs(self.actions[:,wheel_indices]), dim=1)
        wheel_term = wheel_term_1 + wheel_term_2 + wheel_term_3

        # print('term_2',term_2)
        # print('term_3',term_3)
        return hip_knee_term + wheel_term

    def _reward_stand_still_vel_penality(self):
        """当命令很小时，机器人不应该有各个方向的速度"""
        # Penalize motion at zero commands
        term_x = 5 * torch.square(self.base_lin_vel[:, 0])
        term_y_z = torch.sum(torch.square(self.base_lin_vel[:, 1:3]), dim=1)
        return (term_x + term_y_z) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_stand_still_base_pos_penality(self):
        """当命令很小时，机器人不应该有各个方向的位置变化"""
        # Penalize motion at zero commands
        diff_x = 5 * torch.square(self.base_pos[:, 0] - self.base_pos_init[:, 0])
        diff_y_z = torch.sum(torch.square(self.base_pos[:, 1:3] - self.base_pos_init[:, 1:3]), dim=1)
        return (diff_x + diff_y_z) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_stand_still_base_pos(self):
        """当命令很小时，机器人不应该有各个方向的位置变化"""
        # Penalize motion at zero commands
        diff = torch.norm(self.base_pos[:, :2] - self.base_pos_init[:, :2], dim=1)
        # return () * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        return torch.exp(-100 * diff) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    # ------------------------- Rewards Unitree Gym --------------------------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        # print(self.commands[0, 2], self.base_height[0])
        if self.reward_scales["base_height"] < 0:
            return torch.abs(self.base_height - self.commands[:, 2])
        else:
            base_height_error = torch.square(self.base_height - self.commands[:, 2])
            return torch.exp(-200 * base_height_error)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, :2]), dim=1) + torch.sum(
            torch.square(self.dof_vel[:, 3:5]), dim=1
        )

    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos[:, :2] - self.dof_pos_limits[:2, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos[:, :2] - self.dof_pos_limits[:2, 1]).clip(
            min=0.0
        )
        out_of_limits += -(self.dof_pos[:, 3:5] - self.dof_pos_limits[3:5, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos[:, 3:5] - self.dof_pos_limits[3:5, 1]).clip(
            min=0.0
        )
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits).clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return 0.8 * torch.exp(-10 * lin_vel_error) + 0.2 * torch.exp(-40 * lin_vel_error)
    
    def _reward_tracking_lin_vel_enhance(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma / 10) - 1

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    # -------------------------------------------------- wl
    def _reward_nominal_state(self):
        # return torch.square(self.theta0[:, 0] - self.theta0[:, 1])
        if self.reward_scales["nominal_state"] < 0:
            return torch.square(self.theta0[:, 0] - self.theta0[:, 1])
        else:
            ang_diff = torch.square(self.theta0[:, 0] - self.theta0[:, 1])
            return torch.exp(-ang_diff / 0.1)

    def _reward_power(self):
        # Penalize torques
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_wheel_adjustment(self):
        # 鼓励使用轮子的滑动克服前后的倾斜，奖励轮速和倾斜方向一致的情况，并要求轮速方向也一致
        incline_x = self.projected_gravity[:, 0]
        # mean velocity
        wheel_x_mean = (self.foot_velocities[:, 0, 0] + self.foot_velocities[:, 1, 0]) / 2
        # 两边轮速不一致的情况，不给奖励
        wheel_x_invalid = (self.foot_velocities[:, 0, 0] * self.foot_velocities[:, 1, 0]) < 0
        wheel_x_mean[wheel_x_invalid] = 0.0
        wheel_x_mean = wheel_x_mean.reshape(-1)
        reward = incline_x * wheel_x_mean > 0
        return reward
    
    def _reward_wheel_vel(self):
        # Penalize dof velocities
        # left_wheel_vel = self.commands[:,0]/2 - self.commands[:,1]
        # right_wheel_vel = self.commands[:,0]/2 + self.commands[:,1]
        # return torch.sum(torch.square(self.dof_vel[:, 2] - left_wheel_vel) + torch.square(self.dof_vel[:, 5]) - right_wheel_vel)
        return torch.sum(torch.square(self.dof_vel[:, [2, 5]]), dim=1)
