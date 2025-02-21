## RecGS: Removing Water Caustic with Recurrent Gaussian Splatting


<p align="center">
    <img src="recgs.gif" alt="Logo" width="100%">
  </a>
</p>

### Installation
Installation generally follows vanilla Gaussian Splatting installation.
```
git clone git@github.com:tyz1030/recgs.git --recursive
```
or
```
git clone https://github.com/tyz1030/recgs.git --recursive
```
Conda environment is the same with vanilla Gaussian Splatting [(original repo)](https://github.com/graphdeco-inria/gaussian-splatting.git)

### data
to do

### Quickstart
```
conda activate gaussian_splatting
python3 train.py -s /data/xxxxxx    # train a vanila 3DGS first
python3 train_recgs.py -s /data/xxxxxx --start_checkpoint output/xxxxxx/chkpnt30000.pth
python3 render_recgs.py -s /data/xxxxxx -m output/xxxxxx
```

### Citation 
[arXiv](https://www.arxiv.org/abs/2407.10318)
```
@ARTICLE{zhang2025recgs,
  author={Zhang, Tianyi and Zhi, Weiming and Meyers, Braden and Durrant, Nelson and Huang, Kaining and Mangelson, Joshua and Barbalata, Corina and Johnson-Roberson, Matthew},
  journal={IEEE Robotics and Automation Letters}, 
  title={RecGS: Removing Water Caustic With Recurrent Gaussian Splatting}, 
  year={2025},
  volume={10},
  number={1},
  pages={668-675},
  keywords={Three-dimensional displays;Cameras;Sea floor;Filtering;Neural radiance field;Lighting;Robot vision systems;Solid modeling;Robots;Rendering (computer graphics);Deep learning for visual perception;marine robotics},
  doi={10.1109/LRA.2024.3511418}}
```
