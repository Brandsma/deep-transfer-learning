# Deep Learning Practical 1

## How to run

To run the ResNet50 model on CIFAR-10 with the adamax optimizer and a dropout rate of 0.3, run:
```bash
python main.py --load_cifar --model_type ResNet50 --optimizer adamax --with_dropout --dropout_rate 0.3 --epochs 200 --batch_size 16
```

To see all possible options run:
```bash
python main.py --help
```

## Dataset

The dataset is [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10)

## Models

- VGG19
- ResNet50
- MobileNet

## Authors

- A. Brandsma
- J. Van Vliet
