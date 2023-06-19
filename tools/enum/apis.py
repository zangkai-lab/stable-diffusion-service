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


def get_sd_tag(version: str) -> str:
    if version == SDVersions.v1_5_BC:
        return "v1.5"
    if version == SDVersions.ANIME:
        return "anime_nai"
    if version == SDVersions.ANIME_ANYTHING:
        return "anime_anything_v3"
    if version == SDVersions.ANIME_HYBRID:
        return "anime_hybrid_v1"
    if version == SDVersions.ANIME_GUOFENG:
        return "anime_guofeng3"
    if version == SDVersions.ANIME_ORANGE:
        return "anime_orange2"
    return version
