from enum import Enum


class APIs(str, Enum):
    SD = "sd"


class SDVersions(str, Enum):
    v1_5_BC = ""
    v1_5 = "v1.5"
    ANIME = "anime"
    ANIME_ANYTHING = "anime_anything"
    ANIME_HYBRID = "anime_hybrid"
    ANIME_GUOFENG = "anime_guofeng"
    ANIME_ORANGE = "anime_orange"
    DREAMLIKE = "dreamlike_v1"


class Samplers(str, Enum):
    DDIM = "ddim"  # from compvis
    PLMS = "plms"  # from compvis

    K_EULER = "k_euler"  # from k-diffusion
    K_EULER_A = "k_euler_a"  # from k-diffusion
