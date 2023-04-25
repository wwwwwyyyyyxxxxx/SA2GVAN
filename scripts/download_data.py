# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import os
import subprocess


def download_and_uncompress(url, output_dir):
    scene_file = os.path.basename(url)
    print(f'Downloading {url} ...')
    if not os.path.exists(scene_file):
        subprocess.run(['wget -c', url])
    subprocess.run(['tar', '-xzf', scene_file])
    subprocess.run(['rm', scene_file])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--dataset', nargs='+', default=['mp3d', 'replica'])
    parser.add_argument('--rir-type', nargs='+', default=['binaural_rirs', 'ambisonic_rirs'])
    args = parser.parse_args()

    for dataset in args.dataset:
        for rir_type in args.rir_type:
            dataset_rir_dir = os.path.join(args.output_dir, rir_type, dataset)
            aws_root_dir = 'http://dl.fbaipublicfiles.com/SoundSpaces/'
            scenes = os.listdir(os.path.join('data/metadata/', dataset))
            for scene in scenes:
                scene_file = os.path.join(aws_root_dir, rir_type, dataset, scene + '.tar.gz')
                if os.path.exists(os.path.join(dataset_rir_dir, scene)):
                    continue
                else:
                    download_and_uncompress(scene_file, args.output_dir)


if __name__ == '__main__':
    main()