import torch
import torch.nn.functional as F
MAX_FLOW = 400




def sequence_loss(flow_preds, flow_gt, valid, cfg):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    flow_gt_length = torch.sum(flow_gt**2, dim=1).sqrt()
    flow_gt_length = flow_gt_length.view(-1)[valid.view(-1)]
    for t in flow_gt_thresholds:
        e = epe[flow_gt_length < t]
        metrics.update({
                f"{t}-th-5px": (e < 5).float().mean().item()
        })


    return flow_loss, metrics



def sequence_loss_daytime(flow_preds, flow_gt, valid, cfg):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'daytime_epe': epe.mean().item(),
        'daytime_1px': (epe < 1).float().mean().item(),
        'daytime_3px': (epe < 3).float().mean().item(),
        'daytime_5px': (epe < 5).float().mean().item(),
    }

    flow_gt_length = torch.sum(flow_gt**2, dim=1).sqrt()
    flow_gt_length = flow_gt_length.view(-1)[valid.view(-1)]
    for t in flow_gt_thresholds:
        e = epe[flow_gt_length < t]
        metrics.update({
                f"{t}-th-5px": (e < 5).float().mean().item()
        })


    return flow_loss, metrics


def sequence_loss_nighttime(flow_preds, flow_gt, valid, cfg):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'nighttime_epe': epe.mean().item(),
        'nighttime_1px': (epe < 1).float().mean().item(),
        'nighttime_3px': (epe < 3).float().mean().item(),
        'nighttime_5px': (epe < 5).float().mean().item(),
    }

    flow_gt_length = torch.sum(flow_gt**2, dim=1).sqrt()
    flow_gt_length = flow_gt_length.view(-1)[valid.view(-1)]
    for t in flow_gt_thresholds:
        e = epe[flow_gt_length < t]
        metrics.update({
                f"{t}-th-5px": (e < 5).float().mean().item()
        })


    return flow_loss, metrics


def sequence_event_loss(flow_preds, flow_gt, valid, cfg):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'event_epe': epe.mean().item(),
        'event_1px': (epe < 1).float().mean().item(),
        'event_3px': (epe < 3).float().mean().item(),
        'event_5px': (epe < 5).float().mean().item(),
    }

    flow_gt_length = torch.sum(flow_gt**2, dim=1).sqrt()
    flow_gt_length = flow_gt_length.view(-1)[valid.view(-1)]
    for t in flow_gt_thresholds:
        e = epe[flow_gt_length < t]
        metrics.update({
                f"{t}-th-5px": (e < 5).float().mean().item()
        })


    return flow_loss, metrics


def flow_consis_loss(Reflect_flow_preds, image_flow_preds):
    """ Loss function defined over sequence of flow predictions """
    # flow_loss = 0.0

    loss_weight = 0.4
    loss = (Reflect_flow_preds - image_flow_preds).abs()
    loss = loss_weight * loss.mean()

    motion_epe = loss_weight * torch.sum((Reflect_flow_preds - image_flow_preds)**2, dim=1).sqrt()
    motion_epe = motion_epe.view(-1)

    metrics = {
        'consis_epe': motion_epe.mean().item(),
        'consis_epe_1px': (motion_epe < 1).float().mean().item(),
        'consis_epe_3px': (motion_epe < 3).float().mean().item(),
        'consis_epe_5px': (motion_epe < 5).float().mean().item(),
    }

    return loss, metrics




def flow_attention_loss(event_flow, image_flow, attention):
    """ Loss function defined over sequence of flow predictions """
    # flow_loss = 0.0

    loss_weight = 0.4
    loss = (event_flow - image_flow).abs() * attention * loss_weight
    loss = loss.mean()

    motion_epe = loss_weight * torch.sum((event_flow - image_flow)**2, dim=1).sqrt()
    motion_epe = motion_epe.view(-1)

    metrics = {
        'attention_consis_epe': motion_epe.mean().item(),
        'attention_consis_epe_1px': (motion_epe < 1).float().mean().item(),
        'attention_consis_epe_3px': (motion_epe < 3).float().mean().item(),
        'attention_consis_epe_5px': (motion_epe < 5).float().mean().item(),
    }

    return loss, metrics




# cost consistency loss
def cost_memory_consist(image_cost_memory, reflect_cost_memory):
    loss_weight = 0.1
    loss = (reflect_cost_memory - image_cost_memory).abs()
    loss = loss_weight * loss.mean()

    metrics = {
        'cost_consis_loss': loss,      
    }

    return loss, metrics

# daytime_cost: 指导作用
def loss_KL_div(night_cost, daytime_cost, reduction='mean'):
    # compute hist.
    n_h = torch.histc(night_cost[-1].detach())
    d_h = torch.histc(daytime_cost[-1].detach())

    n_h = n_h / torch.sum(n_h)
    d_h = d_h / torch.sum(d_h)

    # log_n = F.log_softmax(n_h)
    # softmax_d = F.softmax(d_h,dim=-1)

    # kl_mean = F.kl_div(log_n, softmax_d, reduction=reduction) * 0.01
    kl_mean = torch.sum(d_h * torch.log(d_h / (n_h+ 1e-8))) * 0.01
    metrics = {
        'kl_div': kl_mean,
    }

    return kl_mean, metrics
