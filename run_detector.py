# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

#
# Run FootAndBall detector on ISSIA-CNR Soccer videos
#

import torch
import time
import cv2
import os
import argparse
import tqdm

import network.footandball as footandball
import data.augmentation as augmentations
from data.augmentation import PLAYER_LABEL, BALL_LABEL
from fps import FPS


def draw_bboxes(image, detections):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if label == PLAYER_LABEL:
            x1, y1, x2, y2 = box
            color = (255, 0, 255)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (int(x1), max(0, int(y1) - 10)), font, 1, color, 2)

        elif label == BALL_LABEL:
            x1, y1, x2, y2 = box
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            color = (0, 200, 255)
            radius = 25
            cv2.circle(image, (int(x), int(y)), radius, color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (max(0, int(x - radius)), max(0, (y - radius - 10))), font, 1,
                        color, 2)

    return image


def run_detector(model, args):
    model.print_summary(show_architecture=False)
    model = model.to(args.device)

    # Initialize FPS tracker
    fps_tracker = FPS()

    if args.device == 'cpu':
        print('Loading CPU weights...')
        state_dict = torch.load(args.weights, map_location=lambda storage, loc: storage)
    else:
        print('Loading GPU weights...')
        state_dict = torch.load(args.weights)

    model.load_state_dict(state_dict)
    # Set model to evaluation mode
    model.eval()

    # Load image "images/0.jpg"
    img = cv2.imread("images/0.jpg")
    img_tensor = augmentations.numpy2tensor(img)
    img_tensor = img_tensor.unsqueeze(dim=0).to(args.device)

    count = 0

    print("Start FPS test")
    fps_tracker.start()
    while True:
        with torch.no_grad():
            # Add dimension for the batch size
            detections = model(img_tensor)[0]

        fps_tracker.update()

        if count % 10 == 0:
            print(fps_tracker.get_fps())

        count += 1


if __name__ == '__main__':
    print('Run FootAndBall detector on input video')

    # Train the DeepBall ball detector model
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name', type=str, default='bohsnet')
    parser.add_argument('--ball_threshold', help='ball confidence detection threshold', type=float, default=0.7)
    parser.add_argument('--player_threshold', help='player confidence detection threshold', type=float, default=0.7)
    parser.add_argument('--device', help='device (CPU or CUDA)', type=str, default='cuda:0')
    args = parser.parse_args()

    if args.model == 'fb1':
        args.weights = "models/model_20201019_1416_final.pth"
    elif args.model == 'bohsnet':
        args.weights = "models/bohsnet_model_12_06_2022_2349_final_with_augs.pth"


    print(f'Model: {args.model}')
    print(f'Model weights path: {args.weights}')
    print(f'Ball confidence detection threshold: {args.ball_threshold}')
    print(f'Player confidence detection threshold: {args.player_threshold}')
    print(f'Device: {args.device}')

    print('')

    assert os.path.exists(
        args.weights
    ), f'Cannot find FootAndBall model weights: {args.weights}'

    model = footandball.model_factory(args.model, 'detect', ball_threshold=args.ball_threshold,
                                      player_threshold=args.player_threshold)

    run_detector(model, args)
