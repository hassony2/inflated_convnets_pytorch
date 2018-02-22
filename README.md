Inflated I3D models with ImageNet weight transfer in PyTorch
============================================================

This repo contains several scripts that allow to inflate 2D networks according to the technique described in 
the paper [*Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset*](https://arxiv.org/abs/1705.07750) by Joao Carreira and Andrew Zisserman to PyTorch.

It provides the inflated versions for :
- ResNet 50, ResNet101, ResNet152
- DenseNet 121, DenseNet161, DenseNet169, DenseNet201

The original (and official!) tensorflow code inflates the inception-v1 network and can be found [here](https://github.com/deepmind/kinetics-i3d/).

So far this code allows for the inflation of DenseNet and ResNet where the basis block is a Bottleneck block (Resnet >50), and the transfer of 2D ImageNet weights.

The 3D network is obtained by going through the layers of the 2D network and inflating them one by one.
The utilities for the inflation (which both inflate the layers and transfer the weights) are located in `src/inflate.py`.

Note that for the ResNet inflation, I use a centered initialization scheme as presented in [*Detect-and-Track: Efficient Pose Estimation in Videos*](https://arxiv.org/abs/1712.09184), where instead of replicating the kernel and scaling the weights by the time dimension (as described in the original I3D paper), I initialize the time-centered slice of the kernel to the 2D weights and the rest to 0.
This allows to obtain (up to numerical differences) the same outputs for the 2D network with the image input and the matching 3D network with 3D inputs (obtained by replicating the 2D image input in the time dimension).

## Use it

To inflate the network and run it on a dummy-dataset with comparison between the final predictions between the original and inflated networks run:

- For ResNet 101 for instance, run `python inflate_resnet.py --resnet_nb 101`  (available for ResNet [50|101|152])

- For DenseNet 121 `python inflate_densenet.py --densenet_nb 121` (available for DenseNet [121|161|169|201])


# Profiling

Forward pass on  GeForce GTX TITAN Black (6Giga) GPU with batch-size 2:

| Network     | time (s)|
|-------------|---------|
| ResNet 50   | 0.6 s  |
| ResNet 101  | 0.8 s  |
| ResNet 152  | 1.1 s  |
| DenseNet 121| 2.6 s  |

Forward pass on  GeForce GTX TITAN Black (6Giga) GPU with batch-size 1:

| Network     | time (s)|
|-------------|---------|
| ResNet 50   |  0.1s   |
| ResNet 101  |  0.3s   |
| ResNet 152  |  0.5s   |
| DenseNet 121| 1.3 s   |
| DenseNet 161| 1.8 s   |
| DenseNet 169| 1.5 s   |
| DenseNet 201| 1.7 s   |

## Note

Another repo with networks pretrained on kinetics is available here [3D-Resnets-Pytorch](https://github.com/kenshohara/3D-ResNets-PyTorch/).
However, it does not transfer the ImageNet weights, which in my experience with inception-v1 did improve the final results.
