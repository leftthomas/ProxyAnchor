# ProxyAnchor

A PyTorch implementation of Proxy Anchor Loss based on CVPR 2020
paper [Proxy Anchor Loss for Deep Metric Learning](https://arxiv.org/abs/2003.13911).

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
```

- pretrainedmodels

```
pip install pretrainedmodels
```

## Datasets

[CARS196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
and [CUB200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
are used in this repo. You should download these datasets by yourself, and extract them into `${data_path}` directory,
make sure the dir names are `car` and `cub`. Then run `data_utils.py` to preprocess them.

## Usage
### Train Model

```
python train.py  --data_name cub --backbone_type inception --feature_dim 256
optional arguments:
--data_path                   datasets path [default value is '/home/data']
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub'])
--backbone_type               backbone network type [default value is 'resnet50'](choices=['resnet50', 'inception', 'googlenet'])
--feature_dim                 feature dim [default value is 512]
--batch_size                  training batch size [default value is 64]
--num_epochs                  training epoch number [default value is 20]
--warm_up                     warm up number [default value is 2]
--recalls                     selected recall [default value is '1,2,4,8']
```

### Test Model

```
python test.py --retrieval_num 10
optional arguments:
--query_img_name              query image name [default value is '/home/data/car/uncropped/008055.jpg']
--data_base                   queried database [default value is 'car_resnet50_512_data_base.pth']
--retrieval_num               retrieval number [default value is 8]
```

## Benchmarks

The models are trained on one NVIDIA GeForce GTX 1070 (8G) GPU. `AdamW` is used to optimize the model, `lr` is `1e-2`
for the parameters of `proxies` and `1e-4` for other parameters, every `5 steps` the `lr` is reduced by `2`.
`weight decay` is used, `scale` is `32` and `margin` is `0.1`, other hyper-parameters are the default values.

### CARS196

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@2</th>
      <th>R@4</th>
      <th>R@8</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">87.2%</td>
      <td align="center">92.4%</td>
      <td align="center">95.5%</td>
      <td align="center">97.4%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1ig6gwBBSm0EPzesL5KytYQ">5bww</a></td>
    </tr>
    <tr>
      <td align="center">Inception</td>
      <td align="center">85.1%</td>
      <td align="center">91.1%</td>
      <td align="center">94.5%</td>
      <td align="center">96.9%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1-wVIlNjiqiUUD1kRh8Efww">r6e7</a></td>
    </tr>
    <tr>
      <td align="center">GoogLeNet</td>
      <td align="center">78.2%</td>
      <td align="center">85.5%</td>
      <td align="center">91.1%</td>
      <td align="center">94.5%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1hMjWx9MG_40oHz6uBqe6OQ">espu</a></td>
    </tr>
  </tbody>
</table>

### CUB200

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@2</th>
      <th>R@4</th>
      <th>R@8</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">67.0%</td>
      <td align="center">77.3%</td>
      <td align="center">85.1%</td>
      <td align="center">90.8%</td>
      <td align="center"><a href="https://pan.baidu.com/s/128SGDlxV1Cd8gPJEi7Z4gA">73h5</a></td>
    </tr>
    <tr>
      <td align="center">Inception</td>
      <td align="center">67.6%</td>
      <td align="center">78.2%</td>
      <td align="center">86.3%</td>
      <td align="center">91.4%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1i97a8vr3Le_9Bk-L0cTJug">u5b9</a></td>
    </tr>
    <tr>
      <td align="center">GoogLeNet</td>
      <td align="center">62.8%</td>
      <td align="center">73.9%</td>
      <td align="center">82.4%</td>
      <td align="center">89.4%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1R6qnPfyBEKysCzWTdnO_6Q">anbq</a></td>
    </tr>
  </tbody>
</table>

## Results

![vis](results/result.png)
