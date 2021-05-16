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
        
        targets = torch.argmax(targets.reshape((n_samples, n_classes, -1)), axis=1)
        inputs = inputs.reshape((n_samples, n_classes, -1))
        
        ce_loss = F.cross_entropy(inputs, targets.type(torch.long), reduction=self.reduction)  # == -log(pt)
        pt = torch.exp(-ce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return f_loss

class BoundingBoxRegressionLoss(torch.nn.Module):
    def __init__(self):
        super(BoundingBoxRegressionLoss, self).__init__()

    def forward(self, inputs, targets, std_preds):
        """
        inputs.shape == targets.shape == (N, 8, RV_WIDTH, RV_HEIGHT)

        std_preds - predicted log standart deviations of bounding box corners
        """

        one_over_sigma = torch.exp(-std_preds).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std_preds = std_preds.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        box_losses = torch.sum(one_over_sigma * (torch.abs(inputs - targets) + std_preds))
        return box_losses


class LaserNetLoss(torch.nn.Module):

    def __init__(self, f_alpha=1, f_gamma=2, focal_loss_reduction='mean'):
        super(LaserNetLoss, self).__init__()

        self.focal_loss = FocalLoss(alpha=f_alpha, gamma=f_gamma, reduction=focal_loss_reduction)
        self.bb_reg_loss = BoundingBoxRegressionLoss()
        self.mse = torch.nn.MSELoss(reduction='mean')

    def forward(self,
                y_pointclass_preds, y_bb_preds, y_std_preds,
                y_pointclass_target, y_bb_target):
        # for now we ignore the bb classification task and do not use bb labels

        L_point_cls = self.focal_loss(inputs=y_pointclass_preds, 
                                      targets=y_pointclass_target)
        
        L_box_corners = self.bb_reg_loss(y_bb_preds, y_bb_target, y_std_preds)

        return torch.mean(L_point_cls + L_box_corners)
