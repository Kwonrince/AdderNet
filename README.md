# AdderNet Reimplementation

CVPR 2020에 발표된 ["AdderNet: Do We Really Need Multiplications in Deep Learning?"](https://arxiv.org/abs/1912.13200)의 구현 코드입니다.
해당 논문은 CNN의 multiplication 연산을 addition 연산으로 바꿔 모바일 기기와 같은 제한된 환경에서 효율적인 연산을 하기 위한 방법을 제시합니다.

논문 리뷰와 구현 결과에 대한 discussion은 [논문리뷰](./slides/AdderNet_review.pdf) 및 [논문리뷰_재현](./slides/AdderNet_final.pdf)에서 확인할 수 있습니다.

## File Description
- `./adder_lenet/` : LeNet-ANN, LeNet-CNN을 MNIST 데이터셋으로 학습
- `./adder_resnet/` : ResNet-ANN, ResNet-CNN을 CIFAR10 데이터셋으로 학습

# Usage
## Requirements
- python 3
- pytorch >= 1.1.0
- torchvision

## LeNet
학습을 위한 MNIST dataset은 기본적으로 `./cache/data/`에 저장됨, `--data`를 통해 변경 가능

모델 저장은 `./cache/data/`에 저장됨, `--output_dir`를 통해 변경 가능

`--mode cnn` : CNN version 학습

`--mode ann` : ANN version 학습

```bash
# usage example
$ cd ./adder_lenet
$ python main.py --mode ann --epoch 100
```

## ResNet
학습을 위한 CIFAR-10 dataset은 기본적으로 `./cache/data/`에 저장됨, `--data`를 통해 변경 가능

모델 저장은 `./cache/data/`에 저장됨, `--output_dir`를 통해 변경 가능

`--mode cnn` : CNN version 학습

`--mode ann` : ANN version 학습


```bash
# usage example
$ cd ./adder_resnet
$ python main.py --mode ann --epoch 50
```

- latency check
```bash
# usage example
$ cd ./adder_resnet
$ python latency.py
```

## Results
학습 결과는 [wandb log](https://wandb.ai/kwonrince/Addernet/overview?workspace=user-kwonrince)에서 확인 가능합니다.
- LeNet
![image](https://user-images.githubusercontent.com/72617445/233326795-606d378b-073b-4725-85be-241220656ba3.png)

- ResNet
![image](https://user-images.githubusercontent.com/72617445/233326984-c336b7e9-7e83-4b91-bf3b-5dfa761d6912.png)

## Reference
[official repository] https://github.com/huawei-noah/AdderNet

[paper] https://arxiv.org/pdf/1912.13200.pdf
