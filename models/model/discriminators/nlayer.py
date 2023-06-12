import torch.nn as nn

from typing import Optional, List

from models.model.discriminators.core import DiscriminatorBase
from models.model.blocks.convs.basic import Conv2d, get_conv_blocks


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@DiscriminatorBase.register("basic")
class NLayerDiscriminator(DiscriminatorBase):
    def __init__(
        self,
        in_channels: int,
        num_classes: Optional[int] = None,
        *,
        num_layers: int = 2,
        start_channels: int = 16,
        norm_type: Optional[str] = "batch",
    ):
        super().__init__(in_channels, num_classes)
        self.num_layers = num_layers
        self.start_channels = start_channels
        # backbone
        blocks: List[nn.Module] = [
            Conv2d(
                in_channels,
                self.start_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        use_bias = norm_type != "batch"
        nc_multiplier = 1
        for i in range(1, num_layers):
            nc_multiplier_prev = nc_multiplier
            nc_multiplier = min(2**i, 8)
            blocks.extend(
                get_conv_blocks(
                    start_channels * nc_multiplier_prev,
                    start_channels * nc_multiplier,
                    4,
                    1 if i == num_layers - 1 else 2,
                    bias=use_bias,
                    padding=1,
                    norm_type=norm_type,
                    activation=nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.net = nn.Sequential(*blocks)
        # heads
        out_channels = start_channels * nc_multiplier
        self.clf = Conv2d(
            out_channels,
            1,
            kernel_size=4,
            padding=1,
            stride=1,
            bias=True,
        )
        # conditional
        self.generate_cond(out_channels)
        # initialize
        self.apply(weights_init)