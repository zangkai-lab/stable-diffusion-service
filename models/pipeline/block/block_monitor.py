from typing import List
from models.pipeline.pipeline_base import Block
from models.config.train_config import DLConfig
from models.model.monitor.monitor_train import TrainerMonitor
from models.model.monitor.monitor_basic import BasicMonitor


@Block.register("build_monitors")
class BuildMonitorsBlock(Block):
    monitors: List[TrainerMonitor]

    def build(self, config: DLConfig) -> None:
        monitor_names = config.monitor_names
        monitor_configs = config.monitor_configs
        if isinstance(monitor_names, str):
            monitor_names = [monitor_names]
        if monitor_names is None:
            self.monitors = [BasicMonitor()]
        else:
            self.monitors = TrainerMonitor.make_multiple(monitor_names, monitor_configs)