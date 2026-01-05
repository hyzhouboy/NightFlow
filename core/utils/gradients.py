import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda'
class Sobel(nn.Module):
    """
    Computes the spatial gradients of 3D data using Sobel filters.
    """

    def __init__(self):
        super().__init__()
        self.pad = nn.ReplicationPad2d(1)
        a = np.zeros((1, 1, 3, 3))
        b = np.zeros((1, 1, 3, 3))
        a[0, :, :, :] = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        b[0, :, :, :] = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.a = torch.from_numpy(a).float().to(DEVICE)
        self.b = torch.from_numpy(b).float().to(DEVICE)

    def forward(self, x):
        """
        :param x: [batch_size x 1 x H x W] input tensor
        :return gradx: [batch_size x 2 x H x W-1] spatial gradient in the x direction
        :return grady: [batch_size x 2 x H-1 x W] spatial gradient in the y direction
        """

        x = x.view(-1, 1, x.shape[2], x.shape[3])  # (batch * channels, 1, height, width)
        x = self.pad(x)
        gradx = F.conv2d(x, self.a, groups=1) / 8  # normalized gradients
        grady = F.conv2d(x, self.b, groups=1) / 8  # normalized gradients
        return gradx, grady



def interpolate(idx, weights, res, polarity_mask=None):
    """
    Create an image-like representation of the warped events.
    :param idx: [batch_size x N x 1] warped event locations
    :param weights: [batch_size x N x 1] interpolation weights for the warped events
    :param res: resolution of the image space
    :param polarity_mask: [batch_size x N x 2] polarity mask for the warped events (default = None)
    :return image of warped events
    """

    if polarity_mask is not None:
        weights = weights * polarity_mask
    iwe = torch.zeros((idx.shape[0], res[0] * res[1], 1)).to(idx.device)
    iwe = iwe.scatter_add_(1, idx.long(), weights)
    iwe = iwe.view((idx.shape[0], 1, res[0], res[1]))
    return iwe

def purge_unfeasible(x, res):
    """
    Purge unfeasible event locations by setting their interpolation weights to zero.
    :param x: location of motion compensated events
    :param res: resolution of the image space
    :return masked indices
    :return mask for interpolation weights
    """

    mask = torch.ones((x.shape[0], x.shape[1], 1)).to(x.device)
    mask_y = (x[:, :, 0:1] < 0) + (x[:, :, 0:1] >= res[0])
    mask_x = (x[:, :, 1:2] < 0) + (x[:, :, 1:2] >= res[1])
    mask[mask_y + mask_x] = 0
    return x * mask, mask


def get_interpolation(events, flow, tref, res, flow_scaling, round_idx=False):
    """
    Warp the input events according to the provided optical flow map and compute the bilinar interpolation
    (or rounding) weights to distribute the events to the closes (integer) locations in the image space.
    :param events: [batch_size x N x 4] input events (y, x, ts, p)
    :param flow: [batch_size x 2 x H x W] optical flow map
    :param tref: reference time toward which events are warped
    :param res: resolution of the image space
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = False)
    :return interpolated event indices
    :return interpolation weights
    """

    # event propagation
    warped_events = events[:, :, 1:3] + (tref - events[:, :, 0:1]) * flow * flow_scaling

    if round_idx:

        # no bilinear interpolation
        idx = torch.round(warped_events)
        weights = torch.ones(idx.shape).to(events.device)

    else:

        # get scattering indices
        top_y = torch.floor(warped_events[:, :, 0:1])
        bot_y = torch.floor(warped_events[:, :, 0:1] + 1)
        left_x = torch.floor(warped_events[:, :, 1:2])
        right_x = torch.floor(warped_events[:, :, 1:2] + 1)

        top_left = torch.cat([top_y, left_x], dim=2)
        top_right = torch.cat([top_y, right_x], dim=2)
        bottom_left = torch.cat([bot_y, left_x], dim=2)
        bottom_right = torch.cat([bot_y, right_x], dim=2)
        idx = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=1)

        # get scattering interpolation weights
        warped_events = torch.cat([warped_events for i in range(4)], dim=1)
        zeros = torch.zeros(warped_events.shape).to(events.device)
        weights = torch.max(zeros, 1 - torch.abs(warped_events - idx))

    # purge unfeasible indices
    idx, mask = purge_unfeasible(idx, res)

    # make unfeasible weights zero
    weights = torch.prod(weights, dim=-1, keepdim=True) * mask  # bilinear interpolation

    # prepare indices
    idx[:, :, 0] *= res[1]  # torch.view is row-major
    idx = torch.sum(idx, dim=2, keepdim=True)

    return idx, weights


class AveragedIWE(nn.Module):
    """
    Returns an image of the per-pixel and per-polarity average number of warped events given
    an optical flow map.
    """

    def __init__(self, config, device):
        super(AveragedIWE, self).__init__()
        self.res = config["loader"]["resolution"]
        self.flow_scaling = max(config["loader"]["resolution"])
        self.batch_size = config["loader"]["batch_size"]
        self.device = device

    def forward(self, flow, event_list, pol_mask):
        """
        :param flow: [batch_size x 2 x H x W] optical flow maps
        :param event_list: [batch_size x N x 4] input events (y, x, ts, p)
        :param pol_mask: [batch_size x N x 2] per-polarity binary mask of the input events
        """

        # original location of events
        idx = event_list[:, :, 1:3].clone()
        idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        idx = torch.sum(idx, dim=2, keepdim=True)

        # flow vector per input event
        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        # get flow for every event in the list
        flow = flow.view(flow.shape[0], 2, -1)
        event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
        event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
        event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
        event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
        event_flow = torch.cat([event_flowy, event_flowx], dim=2)

        # interpolate forward
        fw_idx, fw_weights = get_interpolation(event_list, event_flow, 1, self.res, self.flow_scaling, round_idx=True)

        # per-polarity image of (forward) warped events
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 1:2])
        if fw_idx.shape[1] == 0:
            return torch.cat([fw_iwe_pos, fw_iwe_neg], dim=1)

        # make sure unfeasible mappings are not considered
        pol_list = event_list[:, :, 3:4].clone()
        pol_list[pol_list < 1] = 0  # negative polarity set to 0
        pol_list[fw_weights == 0] = 2  # fake polarity to detect unfeasible mappings

        # encode unique ID for pixel location mapping (idx <-> fw_idx = m_idx)
        m_idx = torch.cat([idx.long(), fw_idx.long()], dim=2)
        m_idx[:, :, 0] *= self.res[0] * self.res[1]
        m_idx = torch.sum(m_idx, dim=2, keepdim=True)

        # encode unique ID for per-polarity pixel location mapping (pol_list <-> m_idx = pm_idx)
        pm_idx = torch.cat([pol_list.long(), m_idx.long()], dim=2)
        pm_idx[:, :, 0] *= (self.res[0] * self.res[1]) ** 2
        pm_idx = torch.sum(pm_idx, dim=2, keepdim=True)

        # number of different pixels locations from where pixels originate during warping
        # this needs to be done per batch as the number of unique indices differs
        fw_iwe_pos_contrib = torch.zeros((flow.shape[0], self.res[0] * self.res[1], 1)).to(self.device)
        fw_iwe_neg_contrib = torch.zeros((flow.shape[0], self.res[0] * self.res[1], 1)).to(self.device)
        for b in range(0, self.batch_size):

            # per-polarity unique mapping combinations
            unique_pm_idx = torch.unique(pm_idx[b, :, :], dim=0)
            unique_pm_idx = torch.cat(
                [
                    unique_pm_idx // ((self.res[0] * self.res[1]) ** 2),
                    unique_pm_idx % ((self.res[0] * self.res[1]) ** 2),
                ],
                dim=1,
            )  # (pol_idx, mapping_idx)
            unique_pm_idx = torch.cat(
                [unique_pm_idx[:, 0:1], unique_pm_idx[:, 1:2] % (self.res[0] * self.res[1])], dim=1
            )  # (pol_idx, fw_idx)
            unique_pm_idx[:, 0] *= self.res[0] * self.res[1]
            unique_pm_idx = torch.sum(unique_pm_idx, dim=1, keepdim=True)

            # per-polarity unique receiving pixels
            unique_pfw_idx, contrib_pfw = torch.unique(unique_pm_idx[:, 0], dim=0, return_counts=True)
            unique_pfw_idx = unique_pfw_idx.view((unique_pfw_idx.shape[0], 1))
            contrib_pfw = contrib_pfw.view((contrib_pfw.shape[0], 1))
            unique_pfw_idx = torch.cat(
                [unique_pfw_idx // (self.res[0] * self.res[1]), unique_pfw_idx % (self.res[0] * self.res[1])],
                dim=1,
            )  # (polarity mask, fw_idx)

            # positive scatter pixel contribution
            mask_pos = unique_pfw_idx[:, 0:1].clone()
            mask_pos[mask_pos == 2] = 0  # remove unfeasible mappings
            b_fw_iwe_pos_contrib = torch.zeros((self.res[0] * self.res[1], 1)).to(self.device)
            b_fw_iwe_pos_contrib = b_fw_iwe_pos_contrib.scatter_add_(
                0, unique_pfw_idx[:, 1:2], mask_pos.float() * contrib_pfw.float()
            )

            # negative scatter pixel contribution
            mask_neg = unique_pfw_idx[:, 0:1].clone()
            mask_neg[mask_neg == 2] = 1  # remove unfeasible mappings
            mask_neg = 1 - mask_neg  # invert polarities
            b_fw_iwe_neg_contrib = torch.zeros((self.res[0] * self.res[1], 1)).to(self.device)
            b_fw_iwe_neg_contrib = b_fw_iwe_neg_contrib.scatter_add_(
                0, unique_pfw_idx[:, 1:2], mask_neg.float() * contrib_pfw.float()
            )

            # store info
            fw_iwe_pos_contrib[b, :, :] = b_fw_iwe_pos_contrib
            fw_iwe_neg_contrib[b, :, :] = b_fw_iwe_neg_contrib

        # average number of warped events per pixel
        fw_iwe_pos_contrib = fw_iwe_pos_contrib.view((flow.shape[0], 1, self.res[0], self.res[1]))
        fw_iwe_neg_contrib = fw_iwe_neg_contrib.view((flow.shape[0], 1, self.res[0], self.res[1]))
        fw_iwe_pos[fw_iwe_pos_contrib > 0] /= fw_iwe_pos_contrib[fw_iwe_pos_contrib > 0]
        fw_iwe_neg[fw_iwe_neg_contrib > 0] /= fw_iwe_neg_contrib[fw_iwe_neg_contrib > 0]

        return torch.cat([fw_iwe_pos, fw_iwe_neg], dim=1)
