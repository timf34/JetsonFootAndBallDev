# Based on code from: https://github.com/lufficc/SSD/
import math

import torch.nn as nn
import torch.nn.functional as F
import torch


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    Args:
        loss (N, n): the loss for each example.
        labels (N, n): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    # Even if there's no positive example for the class, generate 1 * neg_pos_ratio negatives
    num_pos[num_pos == 0] = 1
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask, torch.sum(num_pos).item()


class SSDLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        super(SSDLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, predictions, targets):
        """Compute classification loss and smooth l1 loss.
         input: confidence, predicted_locations, labels, gt_locations
        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """

        ball_conf = predictions
        ball_conf_t = targets
        
        ball_conf_t = torch.stack(ball_conf_t)
       
        batch_size = ball_conf.shape[0]

        # Reshape input tensors
        # ball_conf to (batch_size, n_ball_cells, 2)
        # ball_conf_t to (batch_size, n_ball_cells)
        
        ball_conf = ball_conf.view(batch_size, -1, 2)
        ball_conf_t = ball_conf_t.view(batch_size, -1)

        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            # loss_player is (batch_siz, n_player_cells) tensor
            loss_ball = -F.log_softmax(ball_conf, dim=2)[:, :, 0]
            mask_ball, num_pos_ball = hard_negative_mining(loss_ball, ball_conf_t, self.neg_pos_ratio)

        ball_conf = ball_conf[mask_ball, :]

        ball_loss_c = F.cross_entropy(ball_conf, ball_conf_t[mask_ball], reduction='sum')

        return ball_loss_c/num_pos_ball