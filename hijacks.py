#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/12

from utils import *


# ↓↓↓ repo\pytorch-grad-cam\pytorch_grad_cam\base_cam.py ↓↓↓

from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def BaseCAM_forward_hijack(self: BaseCAM, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False) -> np.ndarray:
  if self.cuda: input_tensor = input_tensor.cuda()

  if self.compute_input_gradient:     # False
    input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

  outputs = self.activations_and_grads(input_tensor)      # <= VRAM consuming
  if targets is None:
    target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
    targets = [ClassifierOutputTarget(category) for category in target_categories]
  
  if self.uses_gradients:     # True
    self.model.zero_grad()
    loss = sum([target(output) for target, output in zip(targets, outputs)])
    loss.backward(retain_graph=True)

  cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
  cam_agg = self.aggregate_multi_layers(cam_per_layer)

  # NOTE: cannot find the vram leak?!!
  self.model.zero_grad()
  self.target_layers.clear()
  self.activations_and_grads.release()
  self.activations_and_grads.handles.clear()
  self.activations_and_grads.activations.clear()
  self.activations_and_grads.gradients.clear()
  gc_everything()

  return cam_agg

# ↑↑↑ repo\pytorch-grad-cam\pytorch_grad_cam\base_cam.py ↑↑↑


# ↓↓↓ nmndeep/robust-segmentation ↓↓↓

from functools import partial

def masked_cross_entropy(pred, target, reduction='none', ignore_index=-1):
    """Cross-entropy of only correctly classified pixels."""

    mask = pred.max(1)[1] == target
    mask = (target != ignore_index) * mask  # TODO: this should be unnecessary.
    loss = F.cross_entropy(pred, target, reduction='none', ignore_index=-1)
    loss = mask.float().detach() * loss
    
    if reduction == 'mean':
        return loss.view(pred.shape[0], -1).mean(-1)
    return loss

def dlr_loss(x, y, reduction='none'):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    return -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

def dlr_loss_targeted(x, y, y_target):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

def segpgd_loss(pred, target, t, max_t, reduction='none', ignore_index=-1):
    """Implementation of the loss of https://arxiv.org/abs/2207.12391.
    pred: B x cls x h x w
    target: B x h x w
    t: current iteration
    max_t: total iterations
    """

    lmbd = t / 2 / max_t
    corrcl = (pred.max(1)[1] == target).float().detach()
    loss = F.cross_entropy(pred, target, reduction='none',
        ignore_index=ignore_index)
    loss = (1 - lmbd) * corrcl * loss + lmbd * (1 - corrcl) * loss

    if reduction == 'mean':
        return loss.view(target.shape[0], -1).mean(-1)
    return loss

def cospgd_loss(pred, target, reduction='mean', ignore_index=-1):
    """Implementation of the loss for semantic segmentation from https://arxiv.org/abs/2302.02213.
    pred: B x cls x h x w
    target: B x h x w
    """

    #with torch.no_grad():
    sigm_pred = torch.sigmoid(pred)
    sh = target.shape
    n_cls = pred.shape[1]
    
    mask_background = (target != ignore_index).long()
    y = mask_background * target  # One-hot encoding doesn't support -1.
    y = F.one_hot(y.view(sh[0], -1), n_cls)
    y = y.permute(0, 2, 1).view(pred.shape)
    #w = (sigm_pred * y).sum(1) / pred.norm(p=2, dim=1) #sigm_pred.max(dim=1)[0] #pred.norm(p=2, dim=1)
    w = F.cosine_similarity(sigm_pred, y)
    w = mask_background * w  # Ignore pixels with label -1.
    
    loss = F.cross_entropy(pred, target, reduction='none', ignore_index=ignore_index)
    #with torch.no_grad():
    assert w.shape == loss.shape
    loss = w.detach() * loss

    if reduction == 'mean':
        return loss.view(sh[0], -1).mean(-1)
    return loss

def margin_loss(pred, target):
    sh = target.shape
    n_cls = pred.shape[1]
    y = F.one_hot(target.view(sh[0], -1), n_cls)
    y = y.permute(0, 2, 1).view(pred.shape)
    logits_target = (y * pred).sum(1)
    logits_other = (pred - 1e10 * y).max(1)[0]

    return logits_other - logits_target

def masked_margin_loss(pred, target):
    """Margin loss of only correctly classified pixels."""

    pred = pred / (pred ** 2).sum(1, keepdim=True).sqrt().detach() #L2_norm(pred, keepdim=True)
    #print(pred.max(), pred.mean())
    #mask = pred.max(1)[1] == target
    loss = margin_loss(pred, target)
    mask = pred.max(1)[1] == target
    #mask = (loss <= 0).detach()
    loss = mask.float().detach() * loss #+ (1 - mask.float()) * torch.log(1 + loss)

    return loss.view(pred.shape[0], -1).mean(-1)

def single_logits_loss(pred, target, normalized=False, reduction='none',
        masked=False, ignore_index=-1):
    """The (normalized) logit of the correct class is minimized."""

    if normalized:
        pred = pred / (pred ** 2).sum(1, keepdim=True).sqrt() #.detach()
    sh = target.shape
    n_cls = pred.shape[1]
    mask_background = (target != ignore_index).long()
    y = target * mask_background  # One-hot doesn't support -1 class.
    y = F.one_hot(y.view(sh[0], -1), n_cls)
    y = y.permute(0, 2, 1).view(pred.shape)
    loss = -1 * (y * pred).sum(1)
    loss = loss * mask_background  # Ignore contribution of background.
    if masked:
        mask = pred.max(1)[1] == target
        loss = mask.float().detach() * loss

    if reduction == 'mean':
        return loss.view(sh[0], -1).mean(-1)
    return loss

def targeted_single_logits_loss(pred, labels, target, normalized=False, reduction='none', masked=False):
    """The (normalized) logit of the target class is maximized."""

    if normalized:
        pred = pred / (pred ** 2).sum(1, keepdim=True).sqrt() #.detach()
    sh = target.shape
    n_cls = pred.shape[1]
    y = F.one_hot(target.view(sh[0], -1), n_cls)
    y = y.permute(0, 2, 1).view(pred.shape)
    loss = (y * pred).sum(1)
    if masked:
        mask = pred.max(1)[1] == labels
        loss = mask.float().detach() * loss

    if reduction == 'mean':
       return loss.view(sh[0], -1).mean(-1)
    return loss

def js_div_fn(p, q, softmax_output=False, reduction='none', red_dim=None, ignore_index=-1):
    """Compute JS divergence between p and q.
    p: logits [bs, n_cls, ...]
    q: labels [bs, ...]
    softmax_output: if softmax has already been applied to p
    reduction: to pass to KL computation
    red_dim: dimensions over which taking the sum
    ignore_index: the pixels with this label are ignored
    """
    
    if not softmax_output: p = F.softmax(p, 1)
    mask_background = (q != ignore_index).long()
    if reduction != 'none' and mask_background.sum() > 0:
        raise ValueError('Incompatible setup.')
    q = mask_background * q  # Change labels -1 to 0 for one-hot.
    q = F.one_hot(q.view(q.shape[0], -1), p.shape[1])
    q = q.permute(0, 2, 1).view(p.shape).float()
    m = (p + q) / 2
    
    loss = (F.kl_div(m.log(), p, reduction=reduction) + F.kl_div(m.log(), q, reduction=reduction)) / 2
    loss = mask_background.unsqueeze(1) * loss  # Ignore contribution of background.
    if red_dim is not None:
        assert reduction == 'none', 'Incompatible setup.'
        loss = loss.sum(dim=red_dim)
        #loss = mask_background * loss
    return loss

def js_loss(p, q, reduction='mean'):
    loss = js_div_fn(p, q, red_dim=(1))  # Sum over classes.
    if reduction == 'mean': return loss.view(p.shape[0], -1).mean(-1)
    elif reduction == 'none': return loss

criterion_dict = {
  'ce': lambda x, y: F.cross_entropy(x, y, reduction='none', ignore_index=-1),
  'ce-avg': lambda x, y: F.cross_entropy(x, y, reduction='none', ignore_index=-1),
  'mask-ce-avg': masked_cross_entropy,
  'js-avg': partial(js_loss, reduction='none'),
  'cospgd-loss': partial(cospgd_loss, reduction='none'),
  'segpgd-loss': partial(segpgd_loss, reduction='none'),
  'margin-avg': lambda x, y: margin_loss(x, y).view(x.shape[0], -1).mean(-1),
  'mask-margin-avg': masked_margin_loss,
  'dlr': dlr_loss,
  'dlr-targeted': dlr_loss_targeted,
  'mask-norm-corrlog-avg': partial(single_logits_loss, normalized=True, reduction='none', masked=True),
  'mask-norm-corrlog-avg-targeted': partial(targeted_single_logits_loss, normalized=True, reduction='none', masked=True),
}

RS_LOSS_DICT = {
  'JS':       criterion_dict['js-avg'],
  'COSPGD':   criterion_dict['cospgd-loss'],
  'SEGPGD':   criterion_dict['segpgd-loss'],
  'CE':       criterion_dict['ce'],
  'CE_MSK':   criterion_dict['mask-ce-avg'],
  'MRG':      criterion_dict['margin-avg'],
  'MRG_MSK':  criterion_dict['mask-margin-avg'],
  'DLR':      criterion_dict['dlr'],
  'DLR_TGT':  criterion_dict['dlr-targeted'],
  'SLGT':     criterion_dict['mask-norm-corrlog-avg'],
  'SLGT_TGT': criterion_dict['mask-norm-corrlog-avg-targeted'],
}

# ↑↑↑ nmndeep/robust-segmentation ↑↑↑
