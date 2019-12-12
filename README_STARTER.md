# CS236DefaultProject

This is the default project for [CS236: Deep Generative Models](https://deepgenerativemodels.github.io/). You'll be implementing text-conditioned image generation.

## Getting Started

First, install the required python libraries as follows. You'll need Python 3.6 or greater.

```
pip -r requirements.txt
```

Next, download the ImageNet32 and class names to the `datasets` directory by executing:

```
bash download_data.sh
```

### Starter Code available

-  A dataloader for CIFAR10 and ImageNet32 in `data.py`.
- Three different types of conditional embeddings for your images in `models/embedders.py`
  - unconditional (conditioning = tensor of zeros for any image)
  - one-hot (one-hot encoding of the image class)
  - text embedding with **BERT** (takes the image caption and embeds it using BERT)
  - The interface "Embedder" makes it easy for you to develop your own text embedding for your specific project.

- The code for a pixelcnn++ model that can take as input an image and a conditional embedding of an image (`models/pixelcnnpp.py`)
- An interface`CaptionConditionedGenerativeModel` that your models should implement if you want to use our training and evaluation script easily. Its main method is the `forward` method that takes as input a batch of images `x`, a batch of conditional (text) embeddings `h` and outputs the loss of your model, in `models/interface.py`.
- Some `utils` that you should check, we use them to train the PixelCNN++ model, they include a function to sample images and display them with the conditioned captions in `utils.py`.
- A training script (`train_pixelcnnpp.py`) that we used to train and evaluate baseline PixelCNN++ models on ImageNet32 and CIFAR10 with non-textual embeddings (unconditional, one-hot). The code makes it easy for you to train and evaluate your own model as long as it implements out interface (models/interface.py)
- We trained a few baseline PixelCNN++ models on CIFAR10 and ImageNet32, you can make use of them if you want to work with PixelCNN++, but we emphasize that they have not been trained with a text embedding of the images:
  - 4 models checkpoints are available in the following drive folder: https://drive.google.com/open?id=1P-ZR4M3xgtXL6gkA7QVqQ07YU-bZ0SvV
    - cifar_conditional.pt -> trained for 90 epochs on CIFAR10's training set, with a one-hot encoding of the class image as the conditional tensor of an image, achieves 3.04 BPDs (bits per dimension) on the validation set.
    - cifar_unconditional.pt -> trained for 89 epochs on CIFAR10's training set, with a constant (zero) conditional embedding of the images, achieves 3.04 BPDs on the validation set.
    - imagenet_conditional.pt ->  trained for 3 epochs on ImageNet32's training set, with a one-hot encoding of the class image as the conditional tensor of an image, achieves 3.73 BPDs on the validation set of ImageNet32.
    - imagenet_unconditional.pt ->  trained for 4 epochs on ImageNet32's training set, constant (zero) conditional embedding of the images, achieves 3.75 BPDs on the validation set of ImageNet32.



