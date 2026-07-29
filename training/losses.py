import torch
import torch.nn as nn
import torch.nn.functional as F

class MorphologyAwareLoss(nn.Module):
    def __init__(self, alpha=0.5, lambda_morph=0.1):
        super(MorphologyAwareLoss, self).__init__()
        self.alpha = alpha
        self.lambda_morph = lambda_morph
        self.cce_loss = nn.CrossEntropyLoss()

    def forward(self, morph_preds, morph_targets, classification_preds, classification_targets, yolo_loss=0.0):
        # L_cce: CrossEntropyLoss on refined classification head
        l_cce = self.cce_loss(classification_preds, classification_targets)
        
        # L_morph: morphology consistency regularization (KL divergence)
        # Using LogSoftmax for preds and Softmax for targets for KLDivLoss
        morph_preds_log = F.log_softmax(morph_preds, dim=-1)
        morph_targets_soft = F.softmax(morph_targets, dim=-1)
        l_morph = F.kl_div(morph_preds_log, morph_targets_soft, reduction='batchmean')
        
        # L_total = L_yolo + alpha*L_cce + lambda*L_morph
        l_total = yolo_loss + self.alpha * l_cce + self.lambda_morph * l_morph
        
        loss_dict = {
            'loss_total': l_total.item(),
            'loss_yolo': yolo_loss.item() if isinstance(yolo_loss, torch.Tensor) else yolo_loss,
            'loss_cce': l_cce.item(),
            'loss_morph': l_morph.item()
        }
        
        return l_total, loss_dict
