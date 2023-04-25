Learning Semantic-Agnostic and Spatial-Aware Representation for Generalizable Visual-Audio Navigation
--------------------------------------------------------------------------------

Pytorch implementation of the RAL 2023 paper [Learning Semantic-Agnostic and Spatial-Aware Representation for Generalizable Visual-Audio Navigation](http://arxiv.org/abs/2304.10773)



Visual-audio navigation (VAN) is attracting more and more attention from the robotic community due to its broad applications, *e.g.*, household robots and rescue robots. In this task, an embodied agent must search for and navigate to the sound source with egocentric visual and audio observations. However, the existing methods are limited in two aspects: 1) poor generalization to unheard sound categories; 2) sample inefficient in training. Focusing on these two problems, we propose a brain-inspired plug-and-play method to learn a semantic-agnostic and spatial-aware representation for generalizable visual-audio navigation. We meticulously design two auxiliary tasks for respectively accelerating learning representations with the above- desired characteristics. With these two auxiliary tasks, the agent learns a spatially-correlated representation of visual and audio inputs that can be applied to work on environments with novel sounds and maps. Experiment results on realistic 3D scenes (Replica and Matterport3D) demonstrate that our method achieves better generalization performance when zero- shot transferred to scenes with unseen maps and unheard sound categories.



![Network16.pdf](./assets/Network16.pdf)



## Dependencies

One can refer to sound-spaces to

You can also use [ez_setup.py](./ez_setup.py) to speed up the downloading.

## Usage
Train and evaluate our models through [run.py](./run.py).

You can substitute {baseline_name} by {av_nav, av_wan}, {environment_name} by {replica, mp3d}.

And You can replace {lambda_classifier} and {lambda_regressor} with the numeric conversion strings of lambda_classifier and lambda_regressor you want, and remove the decimal point.

Note that integers need to add 0 after the decimal point and then remove the decimal point. For example, if you want to call lambda_regressor=1, you need to replace {lambda_regressor} with {10}

1. Training
```
python ss_baselines/{baseline_name}/run.py --exp-config ss_baselines/av_nav/config/audionav/replica/train/audiogoal_depth_classifier_{lambda_classifier}_reg_{lambda_regressor}.yaml --model-dir data/models/{baseline_name}/{environment_name}/audiogoal_depth_classifier_{lambda_classifier}_reg_{lambda_regressor}
```
2. Validation (evaluate each checkpoint and generate a validation curve)
```
python ss_baselines/{baseline_name}/run.py --run-type eval --exp-config ss_baselines/av_nav/config/audionav/replica/val/audiogoal_depth_classifier_{lambda_classifier}_reg_{lambda_regressor}.yaml --model-dir data/models/{baseline_name}/{environment_name}/audiogoal_depth_classifier_{lambda_classifier}_reg_{lambda_regressor}
```
3. Test the best validation checkpoint based on validation curve
```
python ss_baselines/{baseline_name}/run.py --run-type eval --exp-config ss_baselines/{baseline-name}/config/audionav/{environment_name}/test/audiogoal_depth_classifier_{lambda_classifier}_reg_{lambda_regressor}.yaml --model-dir data/models/{environment_name}/audiogoal_depth_classifier_{lambda_classifier}_reg_{lambda_regressor} --model-dir data/models/{baseline_name}/{environment_name} --eval-best
```
4. Test all checkpoints

```
python ss_baselines/{baseline_name}/run.py --run-type eval --exp-config ss_baselines/{baseline-name}/config/audionav/{environment_name}/test/audiogoal_depth_classifier_{lambda_classifier}_reg_{lambda_regressor}.yaml --model-dir data/models/{environment_name}/audiogoal_depth_classifier_{lambda_classifier}_reg_{lambda_regressor} --model-dir data/models/{baseline_name}/{environment_name}
```

## License
SoundSpaces is CC-BY-4.0 licensed, as found in the [LICENSE](LICENSE) file.

The trained models and the task datasets are considered data derived from the correspondent scene datasets.
- Matterport3D based task datasets and trained models are distributed with [Matterport3D Terms of Usehttp://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
- Replica based task datasets, the code for generating such datasets, and trained models are under [Replica license](https://github.com/facebookresearch/Replica-Dataset/blob/master/LICENSE).

