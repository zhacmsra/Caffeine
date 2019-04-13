
PIC=examples/images/cat.jpg
#FLAG=exec
if [ ! -z $2 ]; then
  PIC=$2;
fi

# rm result.txt
export OPENBLAS_NUM_THREADS=32

./build/examples/cpp_classification/classification.bin ./models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt ./models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt $PIC

