from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import argparse
import os
import sys
from keras.utils import plot_model
from ..utils.keras_version import check_keras_version
import keras
import tensorflow as tf



from torch.autograd import Variable
from ..models.resnet import custom_objects


global best_acc

# def finetune(model_weights, best_acc, epochs, lr):
#     optimizer = optim.SGD(model_weights.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
#     for epoch in range(1,epochs):
#         train(model_weights, epoch, optimizer, train_loader)
#         best_acc = test(model_name, model_weights, epoch, test_loader, best_acc)
#     return best_acc

# def deep_compress(models):
#     for model_name, model_weights in models.items():
#         base_model_name = model_name
#         for sparsity in [50.,60.,70.,80.,90.]:
#             # load the pretrained model
#             model_name, model_weights = load_best(model_name, model_weights)
#             model_name = model_name + str(sparsity)
#
#             best_acc = 0.
#
#             # sparsify
#             model_weights = sparsify(model_weights, sparsity)
#
#             # train with 0.01
#             best_acc = finetune(model_weights, best_acc, 30, 0.01)
#             # train with 0.001
#             best_acc = finetune(model_weights, best_acc, 30, 0.001)
#
#         new_model = compress_convs(model_weights, compressed_models[base_model_name])
#
#         # finetune again - this is just to save the model
#         finetune(new_model, 0., 10, 0.001)

# def prune(model, compressed, dims):
#     layers = expand_model(model, [])
#     prunable_layers = []
#
#     for i,layer in enumerate(layers):
#         if isinstance(layer, Conv2D):
#             if layer.prunable:
#                 prunable_layers.append(i)
#
#     ##### Get the number of channels at each layer
#     channels = []
#     for i, layer in enumerate(layers):
#         if i in prunable_layers:
#             c = compress_resnet_conv(i, layers, dims)
#             channels.append(c)
#         else:
#             if isinstance(layer, nn.Conv2d):
#                 channels.append(layer.out_channels)
#
#     print(channels)
#
#     # Init the compressed model
#     compressed_model = compressed(channels)
#
#     ##### Transfer the weights from the original to the compressed model
#
#     for original, compressed in zip(layers, expand_model(compressed_model, [])):
#
#         classes_to_avoid = [nn.Sequential, nn.ReLU, nn.MaxPool2d]
#         has_weight = reduce((lambda b1, b2: b1 and b2), map(lambda c: not isinstance(original, c), classes_to_avoid))
#
#         if has_weight:
#             if original.weight is not None:
#                 compressed.weight.data = original.weight.data
#             if original.bias is not None:
#                 compressed.bias.data   = original.bias.data
#
#     return model

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    parser.add_argument('model',             help='Path to RetinaNet model.')
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',   help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',  help='Max Detections per image (defaults to 100).', default=1, type=int)
    parser.add_argument('--save-path',       help='Path for saving images with detections.')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # load the model
    print('Loading model, this may take a second...')
    model = keras.models.load_model(args.model, custom_objects=custom_objects)
#     plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # print model summary
    print(model.summary())


