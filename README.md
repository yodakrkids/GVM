<div align="center">
<div style="text-align: center;">
    <h1>Generative Video Matting</h1>
</div>

**SIGGRAPH2025**

<p align="center">
<a href='https://yongtaoge.github.io/project/gvm'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://dl.acm.org/doi/10.1145/3721238.3730642"><img src="https://img.shields.io/badge/arXiv-2508.05639-b31b1b.svg"></a> &nbsp;
<a href="https://github.com/aim-uofa/GVM"><img src="https://img.shields.io/badge/GitHub-Code-black?logo=github"></a> &nbsp;
<a href='https://huggingface.co/datasets/geyongtao/video_matting'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a> &nbsp;
<a href="https://huggingface.co/geyongtao/gvm"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>
</p>

</div>


##  📖 Table of Contents

- [Generative Video Matting](#-generative-video-matting)
  - [🔥 News](#-news)
  - [🚀 Getting Started](#-getting-started)
    - [Environment Requirement 🌍](#environment-requirement-)
    - [Download Model Weights ⬇️](#download-️model-weights-)
  - [🏃🏼 Run](#-run)
    - [Inference 📜](#inference-)
    - [Evaluation 📏](#evaluation-)
  - [🎫 License](#-license)
  - [📢 Disclaimer](#-disclaimer)
  - [🤝 Cite Us](#-cite-us)

## 🔥 News
- **August 10, 2025:** Release the inference code and model checkpoints.
- **June 11, 2025:** Repo created. The code and dataset for this project are currently being prepared for release and will be available here soon. Please stay tuned!


## 🚀 Getting Started

### Environment Requirement 🌍

First, clone the repo:

```
git clone https://github.com/aim-uofa/GVM.git
cd GVM
```

Then, we recommend you first use `conda` to create virtual environment, and install needed libraries. For example:

```
conda create -n gvm python=3.10 -y
conda activate gvm
pip install -r requirements.txt
python setup.py develop
```

### Download Model Weights ⬇️

You need to download the model weights by:

```
hugginface-cli download geyongtao/gvm --local-dir data/weights
```

The ckpt structure should be like:

```
|-- GVM    
    |-- data
        |-- weights
            |-- vae
                |-- config.json
                |-- diffusion_pytorch_model.safetensors
            |-- unet
                |-- config.json
                |-- diffusion_pytorch_model.safetensors
            |-- scheduler
                |-- scheduler_config.json  
        |-- datasets
        |-- demo_videos
```



## 🏃🏼 Run

### Inference 📜

You can run generative video matting with:

```
python demo.py \
--model_base 'data/weights/' \
--unet_base data/weights/unet \
--lora_base data/weights/unet \
--mode 'matte' \
--num_frames_per_batch 8 \
--num_interp_frames 1 \
--num_overlap_frames 1 \
--denoise_steps 1 \
--decode_chunk_size 8 \
--max_resolution 960 \
--pretrain_type 'svd' \
--data_dir 'data/demo_videos/xxx.mp4' \
--output_dir 'output_path'
```


### Evaluation 📏

```
TODO
```


## 🎫 License

For academic usage, this project is licensed under [the 2-clause BSD License](LICENSE). For commercial inquiries, please contact [Chunhua Shen](mailto:chhshen@gmail.com).


## 📢 Disclaimer

This repository provides a one-step model for faster inference speed. Its performance is slightly different from the results reported in the original SIGRRAPH paper.

## 🤝 Cite Us

If you find this work helpful for your research, please cite:
```
@inproceedings{ge2025gvm,
author = {Ge, Yongtao and Xie, Kangyang and Xu, Guangkai and Ke, Li and Liu, Mingyu and Huang, Longtao and Xue, Hui and Chen, Hao and Shen, Chunhua},
title = {Generative Video Matting},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3721238.3730642},
doi = {10.1145/3721238.3730642},
booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
series = {SIGGRAPH Conference Papers '25}
}
```