# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, WHEEL_LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
# from .diablo.diablo import Diablo
# from .diablo.diablo_config import DiabloCfg, DiabloCfgPPO

# from .diablo_est.diablo_est_config import DiabloCfg_EST, DiabloCfgPPO_EST

# from .diablo_asm.diablo_asm import DiabloASM
# from .diablo_asm.diablo_asm_config import DiabloASMCfg, DiabloASMCfgPPO
# from .diablo_vmc.diablo_vmc import DiabloVMC
# from .diablo_vmc_step.diablo_vmc_step_config import DiabloVMCStepCfg, DiabloVMCStepCfgPPO
# from .wheel_legged.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO

# from .wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
# from .wheel_legged_vmc.wheel_legged_vmc_config import (
#     WheelLeggedVMCCfg,
#     WheelLeggedVMCCfgPPO,
# )
# from .wheel_legged_vmc_flat.wheel_legged_vmc_flat_config import (
#     WheelLeggedVMCFlatCfg,
#     WheelLeggedVMCFlatCfgPPO,
# )
# from .diablo_vmc.diablo_vmc_config import (
#     DiabloVMCCfg,
#     DiabloVMCCfgPPO,
# )
# from .diablo_vmc_flat.diablo_vmc_flat_config import (
#     DiabloVMCFlatCfg,
#     DiabloVMCFlatCfgPPO,
# )

from .cowa.cowa_env import CowaFreeEnv
from .cowa.cowa_config import CowaCfg, CowaCfgPPO
from .cowa_rma.cowa_rma_config import CowaCfg_RMA, CowaCfgPPO_RMA
from .cowa_vae.cowa_vae_config import CowaCfg_VAE, CowaCfgPPO_VAE
from .cowa_est.cowa_est_config import CowaCfg_EST, CowaCfgPPO_EST

from .cowa_fix.cowa_fix_env import CowaFixEnv
from .cowa_fix.cowa_fix_config import CowaCfg_Fix, CowaCfgPPO_Fix
import os

from wheel_legged_gym.utils.task_registry import task_registry

task_registry.register(
    "cowa", CowaFreeEnv, CowaCfg(), CowaCfgPPO()
)
task_registry.register(
    "cowa_rma", CowaFreeEnv, CowaCfg_RMA(), CowaCfgPPO_RMA()
)
task_registry.register(
    "cowa_vae", CowaFreeEnv, CowaCfg_VAE(), CowaCfgPPO_VAE()
)
task_registry.register(
    "cowa_est", CowaFreeEnv, CowaCfg_EST(), CowaCfgPPO_EST()
)
task_registry.register(
    "cowa_fix", CowaFixEnv, CowaCfg_Fix(), CowaCfgPPO_Fix()
)

