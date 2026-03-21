import torch
from torch import nn

from .utils import PFNLayer, get_paddings_indicator

class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 mode='max',
                 legacy=True,
                 with_doppler_cluster=False,
                 doppler_index=4,
                 doppler_num_bins=4,
                 doppler_temp=4.0,
                 with_doppler_magnitude=False,
                 with_doppler_sign=False):
        super().__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        self.raw_in_channels = in_channels
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 2
        if with_distance:
            in_channels += 1

        # Add a per-pillar Doppler clustering descriptor as point-wise features.
        self._with_doppler_cluster = with_doppler_cluster
        self.doppler_index = doppler_index
        self.doppler_num_bins = doppler_num_bins
        self.doppler_temp = doppler_temp
        self._with_doppler_magnitude = with_doppler_magnitude
        self._with_doppler_sign = with_doppler_sign
        if self._with_doppler_magnitude:
            in_channels += 1
        if self._with_doppler_sign:
            in_channels += 1
        if self._with_doppler_cluster:
            in_channels += 2 * doppler_num_bins
            self.doppler_proto = nn.Parameter(
                torch.linspace(-2.0, 2.0, doppler_num_bins))

        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    def _doppler_cluster_embed(self, raw_features, num_points):
        """Build per-pillar Doppler cluster stats and repeat over points."""
        _, voxel_count, _ = raw_features.shape

        if self.doppler_index < 0 or self.doppler_index >= self.raw_in_channels:
            raise ValueError(
                f'doppler_index={self.doppler_index} out of range for '
                f'raw in_channels={self.raw_in_channels}')

        doppler = raw_features[:, :, self.doppler_index]
        valid = get_paddings_indicator(num_points, voxel_count, axis=0)

        proto = self.doppler_proto.to(dtype=raw_features.dtype,
                                      device=raw_features.device)
        proto = proto.view(1, 1, self.doppler_num_bins)
        doppler_exp = doppler.unsqueeze(-1)

        logits = -self.doppler_temp * (doppler_exp - proto).pow(2)
        logits = logits.masked_fill(~valid.unsqueeze(-1), -1e9)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * valid.unsqueeze(-1).type_as(raw_features)

        denom = weights.sum(dim=1).clamp_min(1e-6)
        mean = (weights * doppler_exp).sum(dim=1) / denom
        var = (weights * (doppler_exp - mean.unsqueeze(1)).pow(2)).sum(dim=1)
        var = var / denom

        doppler_embed = torch.cat([mean, var], dim=-1)
        doppler_embed = doppler_embed.unsqueeze(1).expand(
            -1, voxel_count, -1).contiguous()
        return doppler_embed

    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        raw_features = features
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :2])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
            else:
                f_center = features[:, :, :2]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        if self._with_doppler_magnitude or self._with_doppler_sign:
            if self.doppler_index < 0 or self.doppler_index >= self.raw_in_channels:
                raise ValueError(
                    f'doppler_index={self.doppler_index} out of range for '
                    f'raw in_channels={self.raw_in_channels}')
            doppler = raw_features[:, :, self.doppler_index:self.doppler_index + 1]
            if self._with_doppler_magnitude:
                features_ls.append(doppler.abs())
            if self._with_doppler_sign:
                features_ls.append(torch.sign(doppler))

        if self._with_doppler_cluster:
            doppler_embed = self._doppler_cluster_embed(raw_features, num_points)
            features_ls.append(doppler_embed)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze(1)