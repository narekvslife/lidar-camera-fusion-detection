import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        n_samples, n_classes, _, _ = inputs.shape
        
        targets = torch.argmax(targets, axis=1)
        ce_loss = F.cross_entropy(inputs, targets.type(torch.long), reduction='none')  # == -log(pt)

        pt = torch.exp(-ce_loss)
        
        f_loss = torch.mean(self.alpha * (1 - pt) ** self.gamma * ce_loss, dim=(1, 2))
        return f_loss

class BoundingBoxRegressionLoss(torch.nn.Module):
    def __init__(self):
        super(BoundingBoxRegressionLoss, self).__init__()

    def forward(self, inputs, targets, log_std_preds, box_mask):
        """
        inputs.shape == targets.shape == (N, 8, RV_WIDTH, RV_HEIGHT)

        std_preds - predicted log standart deviations of bounding box corners
        """
        
        box_counts = torch.sum(box_mask, axis=(1, 2, 3))
        box_counts[box_counts == 0 ] = 1
        
        if len(inputs.shape) == 0:
            return torch.tensor(0)

        log_std_preds = log_std_preds.unsqueeze(1)
        one_over_sigma = torch.exp(-log_std_preds)

        box_losses = one_over_sigma * torch.abs(inputs - targets) + log_std_preds  # N x C x W x H

        box_masked_losses = torch.zeros_like(box_losses).masked_scatter(box_mask, box_losses)
        
        return torch.sum(box_masked_losses, axis=(1, 2, 3))  / box_counts


class LaserNetLoss(torch.nn.Module):

    def __init__(self, f_alpha=1, f_gamma=2, focal_loss_reduction='mean'):
        super(LaserNetLoss, self).__init__()

        self.focal_loss = FocalLoss(alpha=f_alpha, gamma=f_gamma, reduction=focal_loss_reduction)
        self.bb_reg_loss = BoundingBoxRegressionLoss()
        self.non_object_labels = [0, 24, 25, 26, 27, 28, 29, 30, 31]

    def forward(self,
                y_pointclass_preds, y_bb_preds, y_logstd_preds, 
                y_pointclass_target, y_bb_target):
        
        
        point_pred_labels = torch.argmax(y_pointclass_target, axis=1)
        
        L_point_cls = self.focal_loss(inputs=y_pointclass_preds,
                                      targets=y_pointclass_target)
        
        bb_mask = (sum([point_pred_labels == nol for nol in self.non_object_labels]) == 0).unsqueeze(1)
        bb_mask = bb_mask.repeat(1, 8, 1, 1)
        
        L_box_corners = self.bb_reg_loss(y_bb_preds,
                                         y_bb_target,
                                         y_logstd_preds,
                                         bb_mask)
        
        print('|point_classification_loss|', L_point_cls.mean().item(), '|bounding_box_loss|', L_box_corners.mean().item())

        return torch.mean(L_point_cls + L_box_corners)
