
from wheel_legged_gym.envs.cowa.cowa_config import CowaCfg, CowaCfgPPO


class CowaCfg_EST(CowaCfg):
    class env(CowaCfg.env):
        frame_stack = 1         #输入给encoder的
        actor_input_stack = 1  #输入给actor的
        num_est_prob = 4        #vel_xyz, height预测的信息的总维度
        c_frame_stack = 2

        num_single_obs = 25 #+ 2 
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs + 12 + 49 + num_est_prob
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

class CowaCfgPPO_EST(CowaCfgPPO):
    runner_class_name = 'OnPolicyRunnerEstimator'
    

    class policy:
        init_noise_std = 1.0
        # estimator para
        estimator_hidden_dims=[128, 64]

    class algorithm(CowaCfgPPO.algorithm):
        # estimator para
        mlp_learning_rate = 5.e-4
        num_adaptation_module_substeps = 1
        
    class runner:
        policy_class_name = 'ActorCritic_Estimator'
        algorithm_class_name = 'PPO_Estimator'
        num_steps_per_env = 48  # per iteration
        max_iterations = 50001  # number of policy updates        #  xxw

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'cowa_est'
        run_name = 'diablo_est'
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = '/home/zhengde.ma/humanoid-gym-main/logs/Cowa_ppo/success/9-4/4-融合了unitree&humanoid'  # updated from load_run and chkpt