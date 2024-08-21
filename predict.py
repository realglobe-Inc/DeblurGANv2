import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
import json
import argparse
from tqdm import tqdm
from pathlib import Path

from aug import get_normalize
from models.networks import get_generator


def fix_relative_path(path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.join('.', os.path.relpath(path, start=Path('.')))


def sorted_glob(patterns):
    files = []
    for pattern in patterns:
        files.extend(glob(pattern, recursive=True))
    return sorted(files)


class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config.yaml',encoding='utf-8') as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)
        model = get_generator(model_name or config['model'])
        model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            inputs = [img.cuda()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]

def process_video(pairs, predictor, output_dir):
    for video_filepath, mask in tqdm(pairs):
        video_filename = os.path.basename(video_filepath)
        output_filepath = os.path.join(output_dir, os.path.splitext(video_filename)[0]+'_deblur.mp4')
        video_in = cv2.VideoCapture(video_filepath)
        fps = video_in.get(cv2.CAP_PROP_FPS)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_num = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        video_out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        tqdm.write(f'process {video_filepath} to {output_filepath}, {fps}fps, resolution: {width}x{height}')
        for frame_num in tqdm(range(total_frame_num), desc=video_filename):
            res, img = video_in.read()
            if not res:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = predictor(img, mask)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            video_out.write(pred)

def main(input_dir='./input/',
         mask_pattern: Optional[str] = None,
         weights_path='checkpoints/fpn_inception.h5',
         output_dir='./output/',
         side_by_side: bool = False,
         video: bool = False):

    img_patterns = [
        os.path.join(os.path.expanduser(input_dir), '**', '*.jpg'),
        os.path.join(os.path.expanduser(input_dir), '**', '*.png')
    ]
    imgs = sorted_glob(img_patterns)
    masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    pairs = zip(imgs, masks)
    names = imgs
    predictor = Predictor(weights_path=weights_path)

    os.makedirs(output_dir, exist_ok=True)
    if not video:
        for name, pair in tqdm(zip(names, pairs), total=len(names)):
            f_img, f_mask = pair

            img = cv2.imread(f_img) if f_img else None
            mask = cv2.imread(f_mask) if f_mask else None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred = predictor(img, mask)
            if side_by_side:
                pred = np.hstack((img, pred))
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

            relative_path = os.path.relpath(f_img, start=Path(input_dir))
            save_path = os.path.join(output_dir, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, pred)
    else:
        process_video(pairs, predictor, output_dir)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=str, default='input', help='Input image or folder'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='output', help='Output folder'
    )
    parser.add_argument(
        '-c', '--checkpoint', type=str, default='checkpoints/fpn_inception.h5', help='Checkpoint Path'
    )

    args = parser.parse_args()
    args.input = os.path.expanduser(fix_relative_path(args.input))
    args.output = os.path.expanduser(fix_relative_path(args.output))
    args.checkpoint = os.path.expanduser(fix_relative_path(args.checkpoint))

    main(input_dir=args.input, weights_path=args.checkpoint, output_dir=args.output)
