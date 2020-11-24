# MP
A PyTorch implementation of Momentum Proxy based on the paper [Momentum proxy based fine-grained image retrieval]().

![Network Architecture](results/structure.png)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```
- pretrainedmodels
```
pip install pretrainedmodels
```
- pytorch-metric-learning
```
conda install pytorch-metric-learning -c metric-learning -c pytorch
```
- SciencePlots
```
pip install SciencePlots
```

## Datasets
[CARS196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) and [CUB200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) 
are used in this repo.

You should download these datasets by yourself, and extract them into `${data_path}` directory, make sure the dir names are 
`car` and `cub`. Then run `data_utils.py` to preprocess them.

## Usage
### Train Model
```
python train.py --momentum 0.6
optional arguments:
--data_path                   datasets path [default value is '/home/data']
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub'])
--backbone_type               backbone network type [default value is 'resnet50'](choices=['resnet50', 'inception', 'googlenet'])
--loss_name                   loss name [default value is 'proxy_nca'](choices=['proxy_nca', 'normalized_softmax', 
                              'cos_face', 'arc_face', 'proxy_anchor'])
--optimizer_type              optimizer type [default value is 'adam*'](choices=['adam*', 'sgd*', 'adam', 'sgd'])
--momentum                    momentum used for the update of moving proxies [default value is 0.5]
--lr                          learning rate [default value is 0.001]
--recalls                     selected recall [default value is '1,2,4,8']
--batch_size                  training batch size [default value is 64]
--num_epochs                  training epoch number [default value is 20]
```

### Test Model
```
python test.py --retrieval_num 10
optional arguments:
--query_img_name              query image name [default value is '/home/data/car/uncropped/008055.jpg']
--data_base                   queried database [default value is 'car_resnet50_proxy_nca_adam*_0.5_20_data_base.pth']
--retrieval_num               retrieval number [default value is 8]
```

## Benchmarks
The models are trained on one NVIDIA GeForce GTX 1070 (8G) GPU. All the hyper-parameters are the default values, and same
with the paper, the results can be downloaded from [BaiduYun(4ahv)](https://pan.baidu.com/s/1CBuOIOXmf_L8kUbIIhuLhw).

### Table

### Figure