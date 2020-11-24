#!/usr/bin/env bash

echo "start experiments......"

python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name proxy_nca --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name proxy_nca --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name proxy_nca --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name proxy_nca --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name normalized_softmax --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name normalized_softmax --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name normalized_softmax --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name normalized_softmax --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name cos_face --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name cos_face --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name cos_face --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name cos_face --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name arc_face --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name arc_face --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name arc_face --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name arc_face --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name proxy_anchor --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name proxy_anchor --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name proxy_anchor --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type resnet50 --loss_name proxy_anchor --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name proxy_nca --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name proxy_nca --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name proxy_nca --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name proxy_nca --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name normalized_softmax --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name normalized_softmax --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name normalized_softmax --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name normalized_softmax --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name cos_face --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name cos_face --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name cos_face --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name cos_face --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name arc_face --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name arc_face --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name arc_face --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name arc_face --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name proxy_anchor --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name proxy_anchor --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name proxy_anchor --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type inception --loss_name proxy_anchor --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name proxy_nca --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name proxy_nca --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name proxy_nca --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name proxy_nca --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name normalized_softmax --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name normalized_softmax --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name normalized_softmax --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name normalized_softmax --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name cos_face --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name cos_face --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name cos_face --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name cos_face --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name arc_face --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name arc_face --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name arc_face --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name arc_face --optimizer_type sgd*
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name proxy_anchor --optimizer_type adam
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name proxy_anchor --optimizer_type sgd
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name proxy_anchor --optimizer_type adam*
wait
python train.py --data_name car --lr 1e-5 --backbone_type googlenet --loss_name proxy_anchor --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name proxy_nca --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name proxy_nca --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name proxy_nca --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name proxy_nca --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name normalized_softmax --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name normalized_softmax --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name normalized_softmax --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name normalized_softmax --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name cos_face --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name cos_face --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name cos_face --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name cos_face --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name arc_face --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name arc_face --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name arc_face --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name arc_face --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name proxy_anchor --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name proxy_anchor --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name proxy_anchor --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type resnet50 --loss_name proxy_anchor --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name proxy_nca --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name proxy_nca --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name proxy_nca --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name proxy_nca --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name normalized_softmax --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name normalized_softmax --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name normalized_softmax --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name normalized_softmax --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name cos_face --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name cos_face --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name cos_face --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name cos_face --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name arc_face --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name arc_face --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name arc_face --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name arc_face --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name proxy_anchor --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name proxy_anchor --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name proxy_anchor --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type inception --loss_name proxy_anchor --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name proxy_nca --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name proxy_nca --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name proxy_nca --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name proxy_nca --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name normalized_softmax --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name normalized_softmax --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name normalized_softmax --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name normalized_softmax --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name cos_face --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name cos_face --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name cos_face --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name cos_face --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name arc_face --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name arc_face --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name arc_face --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name arc_face --optimizer_type sgd*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name proxy_anchor --optimizer_type adam
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name proxy_anchor --optimizer_type sgd
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name proxy_anchor --optimizer_type adam*
wait
python train.py --data_name cub --lr 1e-3 --backbone_type googlenet --loss_name proxy_anchor --optimizer_type sgd*

echo "end experiments......"
