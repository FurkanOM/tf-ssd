# Single Shot MultiBox Detector

Tensorflow SSD implementation from scratch. [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) and [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16) backbones are supported.

It's implemented and tested with **tensorflow 2.0, 2.1, and 2.2**

## Usage

Project models created in virtual environment using [miniconda](https://docs.conda.io/en/latest/miniconda.html).
You can also create required virtual environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

To create virtual environment (tensorflow-2 gpu environment):

```sh
conda env create -f environment.yml
```

There are two different backbone, first one the legacy **vgg16** backbone and the second and default one is **mobilenet_v2**.
You can easily specify the backbone to be used with the **--backbone** parameter.

To train and test SSD model:

```sh
python trainer.py --backbone mobilenet_v2
python predictor.py --backbone vgg16
```

If you have GPU issues you can use **-handle-gpu** flag with these commands:

```sh
python trainer.py -handle-gpu
```

## Trained Model and Examples

Trained with VOC 0712 trainval

Training and validation loss with MobileNetV2 backbone:

![Training and validation loss](http://furkanomerustaoglu.com/wp-content/uploads/2020/04/epoch_loss.png)

Model weights:

* [ssd_mobilenet_v2_model_weights.h5](https://drive.google.com/open?id=1dLhuqIx9HoOtPSqCQlhbts7CjlXa5lCs) (~35 MB)
* [ssd_vgg16_model_weights.h5](https://drive.google.com/open?id=1M-kvXTpwIguTfUVxFpt4DTMRuVhgumyJ) (~105 MB)

| Trained with VOC 0712 trainval data with VGG16 backbone | Trained with VOC 0712 trainval data with MobileNetV2 backbone |
| -------------- | -------------- |
| ![Man with a motorbike](http://furkanomerustaoglu.com/wp-content/uploads/2020/04/man_motorbike.png) | ![Horses](http://furkanomerustaoglu.com/wp-content/uploads/2020/04/ssd_mobilenet_v2_horses.png) |
| Photo by Harley-Davidson on Unsplash | Photo by Mark Neal on Unsplash |
| ![Man with a dog](http://furkanomerustaoglu.com/wp-content/uploads/2020/04/man_dog_cars.png) | ![Airplanes](http://furkanomerustaoglu.com/wp-content/uploads/2020/04/ssd_mobilenet_v2_air_planes.png) |
| Photo by Sebastian Herrmann on Unsplash | Photo by Vishu Gowda on Unsplash |

### References

* VOC 2007 Dataset [[dataset]](http://www.pascal-network.org/challenges/VOC/voc2007/workshop/index.html)
* VOC 2012 Dataset [[dataset]](http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html)
* SSD: Single Shot MultiBox Detector [[paper]](https://arxiv.org/abs/1512.02325)
* MobileNetV2: Inverted Residuals and Linear Bottlenecks [[paper]](https://arxiv.org/abs/1801.04381)
* Original caffe implementation [[code]](https://github.com/weiliu89/caffe/tree/ssd)
* ParseNet: Looking Wider to See Better [[paper]](https://arxiv.org/abs/1506.04579)
* AutoAugment: Learning Augmentation Policies from Data [[paper]](https://arxiv.org/abs/1805.09501)
* Data Augmentation Steps for SSD [[blog]](http://www.telesens.co/2018/06/28/data-augmentation-in-ssd/#Data_Augmentation_Steps)
