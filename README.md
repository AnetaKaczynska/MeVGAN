# MeVGAN
Implementation of Memory Efficient Video GAN.

![Image 1](images/1.png?raw=true)
![Image 2](images/2.png?raw=true)
![Image 3](images/3.png?raw=true)
![Image 4](images/4.png?raw=true)
![Image 5](images/5.png?raw=true)

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
Copy model files from progan/models to MeVGAN/models to recreate original directory structure used to save ProGAN checkpoint. This is necessary to load model weigths.
```
cp -r models/* ../models
```
Prepare custom dataset. Place all your frames in one directory, following the naming convention:
```
<video_name>_<frame_number>.jpg
```

# Datasets

We have used publicly available UCF-101 dataset, which can be found here: https://www.crcv.ucf.edu/data/UCF101.php

Colonoscopy data cannot be publicly shared, but we allow datasets to be available upon request.
For further informaction please contact Tomasz Urbańczyk at tomasz.urbanczyk@dmt.com.pl

# Train

Train FrameSeedGenerator and VideoDiscriminator with pre-trained ProGAN.
```
python train.py -d <path_to_dataset>
```
