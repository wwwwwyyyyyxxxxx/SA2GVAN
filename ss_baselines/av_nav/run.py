import argparse
import logging
import os

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import torch
import tensorflow as tf
import soundspaces
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.av_nav.config.default import get_config


def find_best_ckpt_idx(event_dir_path, min_step=-1, max_step=10000):
    events = os.listdir(event_dir_path)

    max_value = 0
    max_index = -1
    for event in events:
        if "events" not in event:
            continue
        iterator = tf.compat.v1.train.summary_iterator(os.path.join(event_dir_path, event))
        for e in iterator:
            if len(e.summary.value) == 0:
                continue
            t = e.summary.value[0].tag
            m = e.step
            if 'spl' not in e.summary.value[0].tag:
                continue
            if 'softspl' in e.summary.value[0].tag:
                continue
            if 'val' not in t:
                continue
            if not min_step <= e.step <= max_step:
                continue
            v = e.summary.value[0].simple_value
            if len(e.summary.value) > 0 and e.summary.value[0].simple_value > max_value:
                max_value = e.summary.value[0].simple_value
                max_index = e.step

    if max_index == -1:
        print('No max index is found in {}'.format(event_dir_path))
    else:
        print('The best index in {} is {}'.format(event_dir_path, max_index))

    return max_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        default='train',
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default='ss_baselines/av_nav/config/audionav/replica/train/train_template.yaml',
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    parser.add_argument(
        "--eval-best",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    parser.add_argument("--moving-source", default=False, action='store_true',
                        help="Modify config options from command line")
    parser.add_argument(
        "--prev-ckpt-ind",
        type=int,
        default=-1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--best-ckpt",
        type=int,
        default=-1,
    )
    args = parser.parse_args()

    if args.eval_best:
        if args.best_ckpt == -1:
            best_ckpt_idx = find_best_ckpt_idx(os.path.join(args.model_dir, 'tb'))
        else:
            best_ckpt_idx = args.best_ckpt
        best_ckpt_path = os.path.join(args.model_dir, 'data', f'ckpt.{best_ckpt_idx}.pth')
        print(f'Evaluating the best checkpoint: {best_ckpt_path}')
        args.opts += ['EVAL_CKPT_PATH_DIR', best_ckpt_path]
    config = get_config(args.exp_config, args.opts, args.model_dir, args.run_type, args.overwrite)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    torch.set_num_threads(1)

    level = logging.DEBUG if config.DEBUG else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    if args.run_type == "train":
        trainer.train()
    elif args.run_type == "eval":
        trainer.eval(args.eval_interval, args.prev_ckpt_ind, config.USE_LAST_CKPT)


if __name__ == "__main__":
    main()
