from enum import Enum


class Samplers(str, Enum):
    DDIM = "ddim"  # from compvis
    PLMS = "plms"  # from compvis

    K_EULER = "k_euler"  # from k-diffusion
    K_EULER_A = "k_euler_a"  # from k-diffusion



