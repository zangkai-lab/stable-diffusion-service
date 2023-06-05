import os
import json

from typing import Dict, List

from tools.utils.sort import sort_dict_by_value

from models.model.constant import SCORES_FILE


def get_scores(folder: str) -> Dict[str, float]:
    scores_path = os.path.join(folder, SCORES_FILE)
    if not os.path.isfile(scores_path):
        return {}
    with open(scores_path, "r") as f:
        return json.load(f)


def get_sorted_checkpoints(checkpoint_folder: str) -> List[str]:
    # better checkpoints will be placed earlier,
    #  which means `checkpoints[0]` is the best checkpoint
    scores = get_scores(checkpoint_folder)
    if not scores:
        return []
    return list(sort_dict_by_value(scores, reverse=True).keys())