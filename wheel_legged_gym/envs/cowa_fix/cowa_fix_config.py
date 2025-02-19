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


from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class CowaCfg_Fix(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 1
        c_frame_stack = 2
        num_single_obs = 25 + 1
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs + 12 + 4 + 2 + 1 + 6 + 6 + 6
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 6
        num_envs = 8192
        episode_length_s = 20  # episode length in seconds
        use_ref_actions = False
        fail_to_terminal_time_s = 0
        num_est_prob = None

        # privileged obs
        # priv_observe_friction = False  # 1
        # priv_observe_base_mass = False # 1
        # priv_observe_restitution = False #1
        # priv_observe_com_displacement = False    # 3
        
        # priv_observe_motor_strength = False  # 6
        # priv_observe_motor_offset = False    # 6 感觉还是不要了比较好
        # priv_observe_gravity = False    # 3
        # priv_observe_measure_heights = False # 187
    class safety:
        # safety factors
        pos_limit = 1
        vel_limit = 1
        torque_limit = 1    #0.85 xxx 1

    class asset(LeggedRobotCfg.asset):
        # file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged/urdf/cowa_wheel_legged_withoutknee.urdf"
        # file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged/urdf/cowa_wheel_legged.urdf"
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/diablo/urdf/diablo_ASM.urdf"        
        name = "diablo"
        offset = 0.
        l1 = 0.14
        l2 = 0.14
        foot_name = "wheel"
        penalize_contacts_on = ["shank", "thigh", "diablo_base_link"]
        terminate_after_contacts_on = ["shank", "thigh", "diablo_base_link"]

        disable_gravity = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False #
        replace_cylinder_with_capsule = True
        fix_base_link = True
        fix_base_link_height = 0.8  # fix the base of the robot at the height
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = True
        static_friction = 0.8
        dynamic_friction = 0.8
        terrain_length = 10.
        terrain_width = 10.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        max_init_terrain_level = 0  # starting curriculum state  wende 改的，本来是10
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        # terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        terrain_proportions = [0, 0, 0, 1., 0., 0]   #wende 改的，本来是[0, 0, 1, 0., 0., 0, 0] 
        restitution = 0.5
        measured_points_x = [
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]  # 0.6m x 1m rectangle (without center line)
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        slope_treshold = 0.65

    # TODO
    class noise:
        add_noise = True
        noise_level = 1    # scales other values

        class noise_scales:
            dof_pos = 0.02
            dof_vel = 1.5 #0.5
            ang_vel = 0.2
            lin_vel = 0.1
            gravity = 0.05
            quat = 0.1
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.15]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        rand_init_dof = False
        rand_init_dof_range = 0.1  # [rad]
        default_joint_angles = {  # target angles when action = 0.0
            # "left_hip_pitch_joint": 0,
            # "left_knee_pitch_joint": 0,
            # "left_wheel_joint": 0,
            # "right_hip_pitch_joint": 0,
            # "right_knee_pitch_joint": 0,
            # "right_wheel_joint": 0            
            "left_fake_hip_joint": 0.0,
            "left_fake_knee_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_fake_hip_joint": 0.0,
            "right_fake_knee_joint": 0.0,
            "right_wheel_joint": 0.0,         
        }


    # TODO control_type,pos_action_scale,vel_action_scale
    class control(LeggedRobotCfg.control):
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {"hip": 20.0, "knee": 25.0, "wheel": 0}  # [N*m/rad]
        damping = {"hip": 0.4, "knee": 0.4, "wheel": 0.22}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5  # 100Hz
        pos_action_scale = 0.5
        vel_action_scale = 10.0

        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = False
        ratio = 0.9

        feedforward_force = 0 # 42.46kg / 2

    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # 500 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10                                  # xxw
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots = False
        push_interval_s = 7
        max_push_vel_xy = 2.0  # 0.2

        action_noise = 0.02 # 0.02
        action_delay = 0.1 # 0.1

        rand_interval_s = 10    ## 
        randomize_rigids_after_start = False     #控制link的friction和restituion的随机化开关
        randomize_friction = False             # xxw True
        friction_range = [0.1, 2.0]
        randomize_base_mass = False #
        # randomize_mass_range = [0.5, 1.5]         # 乘负载
        added_mass_range = [-2, 3]              # 加负载
        randomize_restitution = False           # TODO
        restitution_range = [0, 1.0]            #加到priv里的东西

        randomize_com_displacement = False      #加到priv里的东西
        com_displacement_range = [-0.05, 0.05]  # base link com的随机化范围
        link_com_displacement_range_factor = 0.2   # link com的随机化比例(与com_displacement_range相乘)

        randomize_inertia = True    
        randomize_inertia_range = [0.8, 1.2]

        randomize_motor_strength = True        #加到priv里的东西
        motor_strength_range = [0.9, 1.1]       #加到priv里的东西

        randomize_PD_factor = True #             
        Kp_factor_range = [0.9, 1.1]            
        Kd_factor_range = [0.9, 1.1]

        randomize_motor_offset = False #
        default_motor_offset = [0,0.0,0,\
                                0,0.0,0]
        motor_offset_range = [-0.03, 0.03]

        randomize_default_dof_pos = True # defautl dof pos位置没变，但数值上有rand的偏差
        randomize_default_dof_pos_range = [-0.05, 0.05]

        gravity_rand_interval_s = 7
        gravity_impulse_duration = 1.0

        randomize_gravity = False # 建议不加
        gravity_range = [-1.0, 1.0]         #

        randomize_lag_timesteps = False      # 模拟delay，对于lag用于给历史的action
        lag_timesteps = 1       #2~4ms walk these ways 加固定action延迟

        randomize_torque_delay = False
        torque_delay_steps = 1

        # randomize_obs_delay = False #用队列加固定obs延迟
        # obs_delay_steps = 1

        # agibot
        add_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [0, 5]   # 0～25ms

        add_dof_lag = True
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 1] # 1~4ms

        add_imu_lag = True # 现在是euler，需要projected gravity                    # 这个是 imu 的延迟
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False         # 不常用always False
        imu_lag_timesteps_range = [0, 1] # 实际10~22ms

        randomize_coulomb_friction = True
        joint_stick_friction_range = [0.1, 0.2]
        joint_coulomb_friction_range = [0.0, 0.0]
        
        randomize_joint_friction = False
        randomize_joint_friction_each_joint = False
        
        default_joint_friction = [0.0, 0.0, 0., 0.0, 0.0, 0.]
        joint_friction_range = [0.8, 1.2]
        # joint_friction_range = [1.5, 1.5]
        joint_1_friction_range = [0.9, 1.1]
        joint_2_friction_range = [0.9, 1.1]
        joint_3_friction_range = [0.9, 1.1]
        joint_4_friction_range = [0.9, 1.1]
        joint_5_friction_range = [0.9, 1.1]
        joint_6_friction_range = [0.9, 1.1]

        randomize_joint_damping = True
        randomize_joint_damping_each_joint = True
        
        default_joint_damping = [0.0, 0.0, 0.0,\
                                 0.0, 0.0, 0.0,]
        joint_damping_range = [0.5, 1.5]
        joint_1_damping_range = [0.8, 1.2]
        joint_2_damping_range = [0.8, 1.2]
        joint_3_damping_range = [0.8, 1.2]
        joint_4_damping_range = [0.8, 1.2]
        joint_5_damping_range = [0.8, 1.2]
        joint_6_damping_range = [0.8, 1.2]

        randomize_joint_armature = False   
        randomize_joint_armature_each_joint = False
        joint_armature_range = [0.001, 0.002]     # Factor
        joint_1_armature_range = [0.001, 0.002]
        joint_2_armature_range = [0.001, 0.002]
        joint_3_armature_range = [0.001, 0.002]
        joint_4_armature_range = [0.001, 0.002]
        joint_5_armature_range = [0.001, 0.002]
        joint_6_armature_range = [0.001, 0.002]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 3
        num_commands = 3
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            height = [0.18, 0.32]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.30         #wh 直立1.1   曲膝 1.06  pos = [0.0, 0.0, 1.36]  # 14 dof版本, 
        only_positive_rewards = False 
        # tracking_sigma = 4    # vel = 0.5 对应 20; vel = 1 对应 4; vel = 1.5 对应 2; vel = 2 对应 1; vel = 3 对应 0.5
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_vel_x = 20
        tracking_sigma_ang_vel = 20
        tracking_vel_enhance = False
        tracking_vel_hard = False    # 非常严苛
        soft_dof_pos_limit = (
            0.97  # percentage of urdf limits, values above this limit are penalized
        )
        max_contact_force = 100  # Forces above this value are penalized xxx 1400
        clip_single_reward = 1


        class scales:
            # feet_contact_forces = -0.05   # xxw -0.01
            # tracking_lin_vel_x = 1.2   # 分开进行vel_x和vel_y奖励计算，本来是3.0，wende改成3.5 ; wh 4
            # # track_vel_hard = 0.6
            # # vel_mismatch_exp = 0.5  # lin_z; ang x,y
            # # low_speed = 0.2     # 0.2 # wh 1
            # default_joint_pos = 0.4   
            # base_acc = 0.3   

            # tracking_lin_vel = 1.0
            # tracking_lin_vel_enhance = 1.0
            # tracking_ang_vel = 1.0

            # base_height = 1.
            nominal_state = -0.1
            # lin_vel_z = -1e-0
            # ang_vel_xy = -0.05
            # orientation = -10.0

            dof_vel = -1e-5
            dof_acc = -1e-5
            torques = -1e-5
            power = -1e-8
            action_rate = -0.1
            action_smoothness = -0.1

            # collision = -20.0
            dof_pos_limits = -0.1
            dof_vel_limits = -0.1
            # torque_limits = -0.1

            # ref sin action
            ref_hip_pos = 2
            ref_wheel_vel = 1

            # wheel_vel = -1e-3
            # wheel_adjustment = 1

            # theta_limit = -0.01
            # same_l = -0.1e-8
            # special for wheel
            # wheel_vel = -0.01


    class normalization:
        class obs_scales:
            lin_vel = 10.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            torques = 0.05
            quat = 1
            gravity = 1

        clip_observations = 100.
        clip_actions = 100.
        


class CowaCfgPPO_Fix(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.0e-4  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 48  # per iteration
        max_iterations = 1000001  # number of policy updates        #  xxw

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'cowa_fix'
        run_name = '学会sin跟随'
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = '/home/zhengde.ma/humanoid-gym-main/logs/Cowa_ppo/success/9-4/4-融合了unitree&humanoid'  # updated from load_run and chkpt
