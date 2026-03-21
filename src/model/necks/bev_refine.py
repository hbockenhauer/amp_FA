from torch import nn


class BEVRefineNeck(nn.Module):
    """Optional lightweight BEV refinement block for ablation studies."""

    def __init__(self, in_channels=384, hidden_channels=384, num_layers=2):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(max(1, num_layers)):
            layers.append(
                nn.Conv2d(
                    c_in,
                    hidden_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False))
            layers.append(nn.BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01))
            layers.append(nn.ReLU(inplace=True))
            c_in = hidden_channels

        self.block = nn.Sequential(*layers)
        self.out_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, feats):
        """Args:
            feats (list[Tensor]): neck outputs, expected single level list.
        """
        if not isinstance(feats, list) or len(feats) != 1:
            return feats
        x = feats[0]
        y = self.block(x)
        y = self.out_proj(y)
        return [x + y]
