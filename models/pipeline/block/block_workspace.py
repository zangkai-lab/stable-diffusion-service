import os
import shutil
from datetime import datetime, timedelta

from models.pipeline.pipeline_base import Block
from models.config.train_config import DLConfig
from models.model.constant import TIME_FORMAT

from tools.mixin.inject_default import InjectDefaultsMixin


def prepare_workspace_from(
    workspace: str,
    *,
    timeout: timedelta = timedelta(30),
    make: bool = True,
) -> str:
    current_time = datetime.now()
    if os.path.isdir(workspace):
        for stuff in os.listdir(workspace):
            if not os.path.isdir(os.path.join(workspace, stuff)):
                continue
            try:
                stuff_time = datetime.strptime(stuff, TIME_FORMAT)
                stuff_delta = current_time - stuff_time
                if stuff_delta > timeout:
                    msg = f"{stuff} will be removed (already {stuff_delta} ago)"
                    print(msg)
                    shutil.rmtree(os.path.join(workspace, stuff))
            except:
                pass
    workspace = os.path.join(workspace, current_time.strftime(TIME_FORMAT))
    if make:
        os.makedirs(workspace)
    return workspace


@Block.register("prepare_workspace")
class PrepareWorkplaceBlock(InjectDefaultsMixin, Block):
    def build(self, config: DLConfig) -> None:
        if not self.is_local_rank_0 or self.training_workspace is None:
            return
        if config.create_sub_workspace:
            workspace = prepare_workspace_from(self.training_workspace)
            config.workspace = workspace
            self._defaults["workspace"] = workspace