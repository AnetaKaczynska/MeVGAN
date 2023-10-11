# MeVGAN
Implementation of Memory Efficient Video GAN.

# Prepare environment

Install requirements.
```
pip install requirements.txt
```
Place ProGAN repo within MeVGAN repo.
```
git clone https://github.com/facebookresearch/pytorch_GAN_zoo progan
```
Apply patch to ProGAN repo.
This will stop visdom session from running in the background, set batch size to 8 for ProGAN training, and add functionality to load ProGAN from checkpoint.
```
cd progan
git apply ../progan.patch
```
Prepare custom dataset. Place all your frames in one directory, following the naming convention:
```
<video_name>_<frame_number>.jpg
```

# Train

Train FrameSeedGenerator and VideoDiscriminator with pre-trained ProGAN.
```
python train.py -d <path_to_dataset>
```
