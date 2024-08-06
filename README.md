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
git clone https://github.com/tyz1030/darkgs.git --recursive
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
@misc{zhang2024recgs,
      title={RecGS: Removing Water Caustic with Recurrent Gaussian Splatting}, 
      author={Tianyi Zhang and Weiming Zhi and Kaining Huang and Joshua Mangelson and Corina Barbalata and Matthew Johnson-Roberson},
      year={2024},
      eprint={2407.10318},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.10318}, 
}
```
