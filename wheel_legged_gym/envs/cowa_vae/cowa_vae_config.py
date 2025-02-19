
from wheel_legged_gym.envs.cowa.cowa_config import CowaCfg, CowaCfgPPO


class CowaCfg_VAE(CowaCfg):
    pass


class CowaCfgPPO_VAE(CowaCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunnerVAE'
    

    class policy(CowaCfgPPO.policy):
        # VAE para
        encoder_hidden_dims=[256, 128]
        decoder_hidden_dims=[64, 128]

    class algorithm(CowaCfgPPO.algorithm):
        # VAE para
        vae_learning_rate = 1.e-3
        kl_weight = 1.
        value_loss_coef = 1.0
        max_grad_norm_vae = 1 # 7e-1
        num_adaptation_module_substeps = 1

    class runner(CowaCfgPPO.runner):
        policy_class_name = 'ActorCritic_VAE'
        algorithm_class_name = 'PPO_VAE'
        num_steps_per_env = 60  # per iteration
        max_iterations = 10001  # number of policy updates        #  xxw

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'cowa_vae'
        run_name = 'v4'
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = '/home/zhengde.ma/humanoid-gym-main/logs/Cowa_ppo/success/9-4/4-融合了unitree&humanoid'  # updated from load_run and chkpt