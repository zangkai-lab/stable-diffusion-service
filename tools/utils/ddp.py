import os

from typing import NamedTuple, Optional


class DDPInfo(NamedTuple):
    rank: int   # global rank
    world_size: int  # number of processes
    local_rank: int  # rank on the current node


def get_ddp_info() -> Optional[DDPInfo]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return DDPInfo(rank, world_size, local_rank)
    return None


def is_local_rank_0() -> bool:
    ddp_info = get_ddp_info()
    return ddp_info is None or ddp_info.local_rank == 0


def get_world_size() -> int:
    ddp_info = get_ddp_info()
    return 1 if ddp_info is None else ddp_info.world_size
