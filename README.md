# PP

A PyTorch implementation of Positive Proxy based on the
paper [Learning with only positive embeddings for fine-grained image retrieval]().

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
python train.py  --data_name cub --backbone_type inception --loss_name normalized_softmax
optional arguments:
--data_path                   datasets path [default value is '/home/data']
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub'])
--backbone_type               backbone network type [default value is 'resnet50'](choices=['resnet50', 'inception', 'googlenet'])
--loss_name                   loss name [default value is 'positive_proxy'](choices=['positive_proxy', 'normalized_softmax'])
--feature_dim                 feature dim [default value is 512]
--batch_size                  training batch size [default value is 64]
--num_epochs                  training epoch number [default value is 20]
--recalls                     selected recall [default value is '1,2,4,8']
```

### Test Model

```
python test.py --retrieval_num 10
optional arguments:
--query_img_name              query image name [default value is '/home/data/car/uncropped/008055.jpg']
--data_base                   queried database [default value is 'car_resnet50_positive_proxy_512_20_data_base.pth']
--retrieval_num               retrieval number [default value is 8]
```

## Benchmarks
The models are trained on one NVIDIA GeForce GTX 1070 (8G) GPU. All the hyper-parameters are the default values, and same
with the paper, the results can be downloaded from [BaiduYun(4ahv)](https://pan.baidu.com/s/1CBuOIOXmf_L8kUbIIhuLhw).

### Table

### Figure