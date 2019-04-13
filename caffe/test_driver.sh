#!/bin/bash

if [ "$1" = "cpu" ] ; then

./build/examples/cpp_classification/classification.bin ./models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt ./models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt list

elif [ "$1" = "fpga" ] ; then

./build/examples/cpp_classification_driverTest/classification_fpga.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt list /home/cdscdemo/Workspace/7v3/myproj/impl/vgg16.xclbin

fi
