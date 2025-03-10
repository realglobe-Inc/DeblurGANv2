import argparse
import os
import traceback
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator


def fix_relative_path(path):
    # pathlibで統一して簡潔にする
    return str(Path(path).expanduser().resolve())


def sorted_glob(patterns):
    files = []
    for pattern in patterns:
        files.extend(glob(pattern, recursive=True))
    return sorted(files)


class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config.yaml', encoding='utf-8') as cfg:
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
        output_filepath = os.path.join(output_dir, os.path.splitext(video_filename)[0] + '_deblur.mp4')
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


def split_and_process_image(img, mask, predictor, tile_size: int, side_by_side=False):
    # 1. 画像の分割処理
    def split_image(img: np.ndarray, tile_size: int) -> tuple[list[tuple[int, int, np.ndarray]], int, int]:
        h, w = img.shape[:2]
        tiles = []
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                tiles.append((x, y, img[y:y + tile_size, x:x + tile_size]))
        return tiles, h, w

    # 2. 分割した画像を統合する関数
    def merge_images(tiles: list[tuple[int, int, np.ndarray]], original_size: tuple[int, int]) -> np.ndarray:
        orig_h, orig_w = original_size
        merged_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for (x, y, tile) in tiles:
            merged_img[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
        return merged_img

    result_tiles = []
    tiles, orig_h, orig_w = split_image(img, tile_size)

    # 各タイルごとに処理する
    for x, y, tile in tiles:
        img_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        m = mask[y:y + tile.shape[0], x:x + tile.shape[1]]
        pred = predictor(img_rgb, m)
        if side_by_side:
            pred = np.hstack((img_rgb, pred))
        processed_tile = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        result_tiles.append((x, y, processed_tile))

    # 3. 処理されたタイルを統合して戻す
    return merge_images(result_tiles, (orig_h, orig_w))


def pad_and_process_image(img, mask, predictor, min_size: int, tile_size: int, side_by_side=False):
    # 処理前の元の画像サイズを保持
    original_h, original_w = img.shape[:2]

    if original_h < min_size or original_w < min_size:
        padded_h = max(original_h, min_size)
        padded_w = max(original_w, min_size)

        # 白色 (255, 255, 255) で画像を埋める
        padded_img = np.ones((padded_h, padded_w, 3), dtype=np.uint8) * 255
        padded_img[:original_h, :original_w] = img
        img = padded_img

        padded_mask = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
        padded_mask[:original_h, :original_w] = mask
        mask = padded_mask

    result_img = split_and_process_image(img, mask, predictor, tile_size, side_by_side)
    # 必要なら元のサイズにトリミング
    return result_img[:original_h, :original_w]


def main(input_dir: str,
         output_dir: str,
         input_format: str,
         output_format: str,
         mask_pattern: Optional[str] = None,
         weights_path='checkpoints/fpn_inception.h5',
         side_by_side: bool = False,
         video: bool = False):
    img_patterns = [
        os.path.join(os.path.expanduser(input_dir), '**', f'*.{input_format}'),
    ]
    imgs = sorted_glob(img_patterns)
    masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    pairs = zip(imgs, masks)
    names = imgs
    predictor = Predictor(weights_path=weights_path)

    skip_count = 0
    os.makedirs(output_dir, exist_ok=True)
    if not video:
        for name, pair in tqdm(zip(names, pairs), total=len(names)):
            f_img, f_mask = pair

            img = cv2.imread(f_img) if f_img else None
            if f_mask:
                mask = cv2.imread(f_mask, cv2.IMREAD_GRAYSCALE)
            else:
                threshold = 254
                white_mask = (img[:, :, 0] >= threshold) & (img[:, :, 1] >= threshold) & (img[:, :, 2] >= threshold)
                # 新しいマスク画像を作成（初期値は黒）
                mask = np.zeros_like(img, dtype=np.uint8)

                # 白でないピクセルを (255,255,255) にする
                mask[~white_mask] = [255, 255, 255]  # 白でない部分を白にする
                # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # # 白に近いピクセルを0、それ以外を255にする
                # _, mask = cv2.threshold(gray_img, 254, 255, cv2.THRESH_BINARY_INV)

            try:
                result_img = pad_and_process_image(img, mask, predictor, 128, 2048, side_by_side)
                # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # pred = predictor(img_rgb, mask)
                # if side_by_side:
                #   pred = np.hstack((img_rgb, pred))
                # result_img = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            except Exception as e:
                skip_count += 1
                print(f"Skipping Debluring {name}: {e}")
                traceback.print_exc()
                result_img = img

            relative_path = os.path.relpath(f_img, start=Path(input_dir))
            save_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + f'.{output_format}')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, result_img)
    else:
        process_video(pairs, predictor, output_dir)

    print(f"Skip : {skip_count}, Success : {len(names) - skip_count}")


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
    parser.add_argument(
        '--input-format', type=str, default='png', help='Input image extension'
    )
    parser.add_argument(
        '--output-format', type=str, default='png', help='Output image extension'
    )

    args = parser.parse_args()
    args.input = fix_relative_path(args.input)
    args.output = fix_relative_path(args.output)
    args.checkpoint = fix_relative_path(args.checkpoint)

    main(input_dir=args.input, weights_path=args.checkpoint, output_dir=args.output, input_format=args.input_format,
         output_format=args.output_format)
