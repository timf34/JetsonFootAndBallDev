import torch
import torch.nn as nn


import os
import sys
sys.path.append(os.path.abspath(''))

import network.bohsnet.fpn as fpn
import network.bohsnet.nms as nms

BALL_LABEL = 1
BALL_DELTA = 3
PLAYER_LABEL = 2
# Size of the ball bbox in pixels (fixed as we detect only ball center)
BALL_BBOX_SIZE = 20


# Get ranges of cells to mark with ground truth location
def get_active_cells(bbox_center_x, bbox_center_y, downsampling_factor, conf_width, conf_height, delta):
    cell_x = int(bbox_center_x / downsampling_factor)
    cell_y = int(bbox_center_y / downsampling_factor)
    x1 = max(cell_x - delta // 2, 0)
    x2 = min(cell_x + delta // 2, conf_width - 1)
    y1 = max(cell_y - delta // 2, 0)
    y2 = min(cell_y + delta // 2, conf_height - 1)
    return x1, y1, x2, y2


def cell2pixel(cell_x, cell_y, downsampling_factor):
    # Inverse function to get_active_cells
    # Returns a range of pixels corresponding to the given cell
    x1 = cell_x * downsampling_factor
    x2 = cell_x * downsampling_factor + downsampling_factor - 1
    y1 = cell_y * downsampling_factor
    y2 = cell_y * downsampling_factor + downsampling_factor - 1
    return x1, y1, x2, y2


def cell_center(cell_x, cell_y, downsampling_factor):
    # Pixel coordinates of the cell center
    x = cell_x * downsampling_factor + (downsampling_factor - 1) / 2
    y = cell_y * downsampling_factor + (downsampling_factor - 1) / 2
    return x, y


def create_groundtruth_maps(bboxes, blabels, img_shape, ball_downsampling_factor, ball_delta):
    # Generate ground truth: player location map, player confidence map and ball confidence map
    # targets: List of ground truth player and ball positions
    # img_shape: shape of the input image
    # ball_delta: number of cells marked around the bbox center (must be an odd number: 1, 3, 5, ....)

    # Number of elements in the minibatch
    num = len(bboxes)

    h, w = img_shape
    # Size of target confidence maps
    ball_conf_height = h // ball_downsampling_factor
    ball_conf_width = w // ball_downsampling_factor

    # match priors (default boxes) and ground truth boxes
    ball_conf_t = torch.zeros([num, ball_conf_height, ball_conf_width], dtype=torch.long)

    for idx, (boxes, labels) in enumerate(zip(bboxes, blabels)):
        # Iterate over all batch elements
        for box, label in zip(boxes, labels):
            # Iterate over all objects in a single frame
            bbox_center_x = (box[0] + box[2]) / 2.
            bbox_center_y = (box[1] + box[3]) / 2.

            # print('bbox_center_x: {}'.format(bbox_center_x), 'bbox_center_y: {}'.format(bbox_center_y))

            if label == BALL_LABEL:
                # Convert bbox centers to cell coordinates in the ball confidence map
                x1, y1, x2, y2 = get_active_cells(bbox_center_x, bbox_center_y, ball_downsampling_factor,
                                                  ball_conf_width, ball_conf_height, ball_delta)

                # print('x1: {}'.format(x1), 'y1: {}'.format(y1), 'x2: {}'.format(x2), 'y2: {}'.format(y2))

                ball_conf_t[idx, y1:y2 + 1, x1:x2 + 1] = 1

    return ball_conf_t


def count_parameters(model):
    # Count number of parameters in the network: all and trainable
    # Return tuple (all_parametes, trainable_parameters)

    if model is None:
        return 0, 0
    else:
        ap = sum(p.numel() for p in model.parameters())
        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return ap, tp


class FootAndBall(nn.Module):
    def __init__(self, phase, base_network: nn.Module, ball_classifier: nn.Module, max_ball_detections=100,
                 ball_threshold=0.0, ball_delta: int=3, ball_downsampling_factor: int=4):
        # phase: in 'train' returns unnormalized confidence values in feature maps (logits)
        #           Note: that now the confidence maps **are** normalized. This was done to overfit on single image
        #        in 'eval' returns normalized confidence values (passed through Softmax)
        #        in 'detect' returns detection bounding boxes
        # max_player_detections, max_ball_detections, player_threshold, ball_threshold: used
        #        only in 'detect' mode
        super(FootAndBall, self).__init__()

        assert phase in ['train', 'eval', 'detect']

        self.phase = phase
        self.base_network = base_network
        self.ball_classifier = ball_classifier
        self.max_ball_detections = max_ball_detections
        self.ball_threshold = ball_threshold
        # Downsampling factor for ball and player feature maps
        self.ball_downsampling_factor = 4
        # Number of cells marked around the bbox center
        # By default we mark 1x1 cells for players (each cell having 16x16 pixels) and 3x3 cells for ball (each cell
        # having 4x4 pixels)
        self.ball_delta = ball_delta

        # Note that this softmax code can be tidied up but we will wait until we have trained the network with softmax
        # and see if it works before we tidy it up/ keep the softmax permananetly during training mode.
        self.softmax = nn.Softmax(dim=1)
        self.nms_kernel_size = (3, 3)
        self.nms = nms.NonMaximaSuppression2d(self.nms_kernel_size)

    def detect_from_map(self, confidence_map, downscale_factor, max_detections, bbox_map=None):
        # downscale_factor: downscaling factor of the confidence map versus an original image
        # Confidence map is [B, C=2, H, W] tensor, where C=0 is background and C=1 is an object

        confidence_map = self.nms(confidence_map)[:, 1, :, :]
        # confidence_map is (B, H, W) tensor
        batch_size, h, w = confidence_map.shape[0], confidence_map.shape[1], confidence_map.shape[2]
        confidence_map = confidence_map.view(batch_size, -1)

        values, indices = torch.sort(confidence_map, dim=-1, descending=True)
        if max_detections < indices.shape[1]:
            indices = indices[:, :max_detections]

        # Compute indexes of cells with detected object
        xc = indices % w
        yc = indices // w

        # Compute pixel coordinates of cell centers
        xc = xc.float() * downscale_factor + (downscale_factor - 1.) / 2.
        yc = yc.float() * downscale_factor + (downscale_factor - 1.) / 2.

        # Bounding boxes are encoded as a relative position of the centre (with respect to the cell centre)
        # and it's width and height in normalized coordinates (where 1 is the width/height of the player
        # feature map)
        # Position x and y of the bbox centre offset in normalized coords
        # (dx, dy, w, h)

        if bbox_map is not None:
            # bbox_map is (B, C=4, H, W) tensor
            bbox_map = bbox_map.view(batch_size, 4, -1)
            # bbox_map is (B, C=4, H*W) tensor
            # Convert from relative to absolute (in pixel) values
            bbox_map[:, 0] *= w * downscale_factor
            bbox_map[:, 2] *= w * downscale_factor
            bbox_map[:, 1] *= h * downscale_factor
            bbox_map[:, 3] *= h * downscale_factor
        else:
            # For the ball, bbox map is not given! Create fixed-size bboxes
            batch_size, h, w = confidence_map.shape[0], confidence_map.shape[-2], confidence_map.shape[-1]
            bbox_map = torch.zeros((batch_size, 4, h * w), dtype=torch.float32).to(confidence_map.device)
            bbox_map[:, [2, 3]] = BALL_BBOX_SIZE

        # Resultant detections (batch_size, max_detections, bbox),
        # where bbox = (x1, y1, x2, y2, confidence) in pixel coordinates
        detections = torch.zeros((batch_size, max_detections, 5), dtype=torch.float32).to(confidence_map.device)

        for n in range(batch_size):
            temp = bbox_map[n, :, indices[n]]
            # temp is (4, n_detections) tensor, with bbox details in pixel units (dx, dy, w, h)
            # where dx, dy is a displacement of the box center relative to the cell center

            # Compute bbox centers = cell center + predicted displacement
            bx = xc[n] + temp[0]
            by = yc[n] + temp[1]

            detections[n, :, 0] = bx - 0.5 * temp[2]  # x1
            detections[n, :, 2] = bx + 0.5 * temp[2]  # x2
            detections[n, :, 1] = by - 0.5 * temp[3]  # y1
            detections[n, :, 3] = by + 0.5 * temp[3]  # y2
            detections[n, :, 4] = values[n, :max_detections]

        return detections

    def detect(self, ball_feature_map):
        """
        args: ball_feature_map: (B, C=2, H, W) tensor
        """
        # downscale_factor: downscaling factor of the confidence map versus an original image

        ball_detections = self.detect_from_map(ball_feature_map, self.ball_downsampling_factor, self.max_ball_detections)

        # Iterate over batch elements and prepare a list with detection results
        output = []
        for ball_det in ball_detections:
            # Filter out detections below the confidence threshold
            ball_det = ball_det[ball_det[..., 4] >= self.ball_threshold]
            ball_boxes = ball_det[..., 0:4] # x1, y1, x2, y2 (slicing from 0 to 4 (not including 4))
            ball_scores = ball_det[..., 4]
            #TODO: note that we are doing detections in torch.int64, it might be worth bringing it down to int32
            ball_labels = torch.tensor([BALL_LABEL] * len(ball_det), dtype=torch.int64)

            # This was just to concatenate the player scores with the ball scores - however, we have removed the former
            boxes = torch.cat([ball_boxes], dim=0)
            scores = torch.cat([ball_scores], dim=0)
            labels = torch.cat([ball_labels], dim=0)

            # TODO: Note that I have changed the output here to include the raw ball feature map
            # temp = {'boxes': boxes, 'labels': labels, 'scores': scores, 'ball_feature_map': ball_feature_map}
            # Removing the ball feature map key, will clean up later 
            temp = {'boxes': boxes, 'labels': labels, 'scores': scores}
            output.append(temp)

        return output

    def groundtruth_maps(self, boxes, labels, img_shape):
        # Generate ground truth: player location map, player confidence map and ball confidence map
        # targets: List of ground truth player and ball positions
        # img_shape: shape of the input image

        ball_conf_t = create_groundtruth_maps(boxes, labels, img_shape, self.ball_downsampling_factor, self.ball_delta)

        return ball_conf_t

    def forward(self, x):
        """
            Ball feature maps returned from training mode are: [B, H, W, C=2]

            Ball feature maps returned as part of a dict from detect mode are: [B, C=2, H, W]
        """


        height, width = x.shape[2], x.shape[3]

        x = self.base_network(x)
        # x must return 2 tensors
        # one (higher spatial resolution) is for ball detection (downsampled by 4)
        # the other (lower spatial resolution) is for players detection (downsampled by 16)

        assert len(x) == 2
        # Same batch size for two tensors
        assert x[0].shape[0] == x[1].shape[0]
        # Same number of channels
        assert x[0].shape[1] == x[1].shape[1]
        # The first has higher spatial resolution then the other
        assert x[0].shape[2] == height // self.ball_downsampling_factor
        assert x[0].shape[3] == width // self.ball_downsampling_factor

        ball_feature_map = self.ball_classifier(x[0])

        # change
        # if self.phase in ['eval', 'detect', 'train']:
        if self.phase in ['eval', 'detect']:
            # In eval and detect mode, convert logits to normalized confidence in [0..1] range
            ball_feature_map = self.softmax(ball_feature_map)

        if self.phase in ['train', 'eval']:
            # Permute dimensions, so channel is the last one (batch_size, h, w, n_channels)!
            ball_feature_map = ball_feature_map.permute(0, 2, 3, 1).contiguous()
            # loc has shape (n_batch_size, feature_map_size_y, feature_map_size_x, 4)
            # conf has shape (n_batch_size, feature_map_size_y, feature_map_size_x, 2)
            output = ball_feature_map
        elif self.phase == 'detect':
            # Detect bounding boxes
            output = self.detect(ball_feature_map)

        return output

    def print_summary(self, show_architecture=True):
        # Print network statistics
        if show_architecture:
            print('Base network:')
            print(self.base_network)
            if self.ball_classifier is not None:
                print('Ball classifier:')
                print(self.ball_classifier)

        ap, tp = count_parameters(self.base_network)
        print(f'Base network parameters (all/trainable): {ap}/{tp}')

        if self.ball_classifier is not None:
            ap, tp = count_parameters(self.ball_classifier)
            print(f'Ball classifier parameters (all/trainable): {ap}/{tp}')

        ap, tp = count_parameters(self)
        print(f'Total (all/trainable): {ap} / {tp}')
        print('')


def build_bohsnet_detector1(phase='train', ball_threshold=0.7, player_threshold=0.7, max_ball_detections=2, max_player_detections=22):
    # phase: 'train' or 'test'
    assert phase in ['train', 'test', 'detect']

    # layers, out_channels = fpn.make_dsc_modules(fpn.cfg['X'], batch_norm=True)
    layers, out_channels = fpn.make_modules(fpn.cfg['X'], batch_norm=True)
    # FPN returns 3 tensors for each input: one dowscaled 4 times in each input dimension, the other downscaled 16 times
    # tensor with 2 channels downscaled 4 times is used for ball detection
    # tensor with 2 channels downscaled 16 times is used for the player detection (1 location corresponds to 16x16 pixel block)
    # tensor with 4 channels downscaled 16 times is used for the player bbox regression
    lateral_channels = 32
    i_channels = 32

    base_net = fpn.FPN(layers, out_channels=out_channels, lateral_channels=lateral_channels, return_layers=[1, 3])
    ball_classifier = nn.Sequential(nn.Conv2d(lateral_channels, out_channels=i_channels, kernel_size=(3, 3), padding=1),
                                    nn.ReLU(inplace=True), nn.Conv2d(i_channels, out_channels=2, kernel_size=(3, 3),
                                                                     padding=1))

    return FootAndBall(phase, base_net, ball_classifier=ball_classifier, ball_threshold=ball_threshold,
                       max_ball_detections=max_ball_detections, ball_delta=BALL_DELTA)


def bohs_model_factory(model_name, phase, max_ball_detections=2, ball_threshold=0.7):
    if model_name == 'bohsnet':
        model_fn = build_bohsnet_detector1
    else:
        print(f'Model not implemented: {model_name}')
        raise NotImplementedError

    return model_fn(phase, ball_threshold=ball_threshold, max_ball_detections=max_ball_detections)


if __name__ == '__main__':
    net = bohs_model_factory('bohsnet', 'train')
    net.print_summary()

    x = torch.zeros((2, 3, 1024, 1024))
    x = net(x)

    # print("x", x)

    for t in x:
        print(t.shape)

    print('.')
