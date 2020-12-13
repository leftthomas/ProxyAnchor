# PP

A PyTorch implementation of Positive Proxy Loss based on the
paper [Positive Proxy Loss for fine-grained image retrieval]().

![Network Architecture](results/structure.png)

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

- AdamP

```
pip install adamp
```

- SciencePlots

```
pip install SciencePlots
```

## Datasets

[CARS196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
and [CUB200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
are used in this repo. You should download these datasets by yourself, and extract them into `${data_path}` directory,
make sure the dir names are `car` and `cub`. Then run `data_utils.py` to preprocess them.

## Usage
### Train Model

```
python train.py  --data_name cub --backbone_type inception --loss_name normalized_softmax
optional arguments:
--data_path                   datasets path [default value is '/home/data']
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub'])
--backbone_type               backbone network type [default value is 'resnet50'](choices=['resnet50', 'inception', 'googlenet'])
--loss_name                   loss name [default value is 'proxy_anchor'](choices=['proxy_anchor', 'normalized_softmax'])
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
--data_base                   queried database [default value is 'car_resnet50_proxy_anchor_512_20_data_base.pth']
--retrieval_num               retrieval number [default value is 8]
```

## Benchmarks

The models are trained on one NVIDIA GeForce GTX 1070 (8G) GPU. `lr` is `1e-2` for the parameters of `ProxyLinear`
and `1e-4` for other parameters, every `5 steps` the `lr` is reduced by `2`.
`scale` is `32` and `margin` is `0.1`, other hyper-parameters are the default values.

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
      <td align="center">87.0%</td>
      <td align="center">92.5%</td>
      <td align="center">95.6%</td>
      <td align="center">97.4%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1buUyFR5bShLcvVkmnB5kEA">4ahv</a></td>
    </tr>
    <tr>
      <td align="center">Inception</td>
      <td align="center">84.0%</td>
      <td align="center">90.5%</td>
      <td align="center">94.3%</td>
      <td align="center">96.6%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1CBuOIOXmf_L8kUbIIhuLhw">w66g</a></td>
    </tr>
    <tr>
      <td align="center">GoogLeNet</td>
      <td align="center">78.8%</td>
      <td align="center">86.2%</td>
      <td align="center">91.5%</td>
      <td align="center">95.1%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1O2l49xOKiAmCP3kPq81npA">i3tz</a></td>
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
      <td align="center">70.0%</td>
      <td align="center">79.6%</td>
      <td align="center">86.9%</td>
      <td align="center">92.2%</td>
      <td align="center"><a href="https://pan.baidu.com/s/11xIyDFbUdjpgMJbXVsZbPw">ek5w</a></td>
    </tr>
    <tr>
      <td align="center">Inception</td>
      <td align="center">66.7%</td>
      <td align="center">77.3%</td>
      <td align="center">85.7%</td>
      <td align="center">91.0%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1Qo9Ax-9HDrkMn0wewvIUOw">48qr</a></td>
    </tr>
    <tr>
      <td align="center">GoogLeNet</td>
      <td align="center">62.5%</td>
      <td align="center">73.6%</td>
      <td align="center">82.8%</td>
      <td align="center">89.8%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1N4e9VcF72T4TQqmciPqWGw">s4uv</a></td>
    </tr>
  </tbody>
</table>

## Results

![vis](results/results.png)