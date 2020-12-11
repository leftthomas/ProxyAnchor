#!/bin/bash

if [ -n "$1" ]; then
  path="$1"
else
  path="/home/data"
fi

if [ -n "$2" ]; then
  size=$2
else
  size=64
fi

if [ -n "$3" ]; then
  epochs=$3
else
  epochs=20
fi

if [ -n "$4" ]; then
  recall="$4"
else
  recall="1,2,4,8"
fi

data_name=("car" "cub")
backbone_type=("resnet50" "inception" "googlenet")
loss_name=("proxy_nca" "normalized_softmax" "cos_face" "arc_face" "proxy_anchor")
optimizer_type=("adamP" "sgdP" "adam" "sgd")

for data in ${data_name[*]}; do
  for backbone in ${backbone_type[*]}; do
    for loss in ${loss_name[*]}; do
      for optimizer in ${optimizer_type[*]}; do
        if [[ ${optimizer} =~ "adam" ]]; then
          lr=2e-5
        else
          lr=2e-3
        fi
          echo "python train.py --data_path ${path} --data_name ${data} --backbone_type ${backbone} --loss_name ${loss} --optimizer_type ${optimizer} --lr ${lr} --batch_size ${size} --num_epochs ${epochs} --recalls ${recall}"
          # shellcheck disable=SC2086
          python train.py --data_path ${path} --data_name ${data} --backbone_type ${backbone} --loss_name ${loss} --optimizer_type ${optimizer} --lr ${lr} --batch_size ${size} --num_epochs ${epochs} --recalls ${recall}
      done
    done
  done
done
