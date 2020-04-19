# Single Shot MultiBox Detector

Tensorflow SSD implementation from scratch.

It's implemented and tested with **tensorflow 2.0, 2.1, and 2.2**

## Usage

Project models created in virtual environment using [miniconda](https://docs.conda.io/en/latest/miniconda.html).
You can also create required virtual environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

To create virtual environment (tensorflow-2 gpu environment):

```sh
conda env create -f environment.yml
```

To train and test SSD model:

```sh
python trainer.py
python predictor.py
```

If you have GPU issues you can use **-handle-gpu** flag with these commands:

```sh
python trainer.py -handle-gpu
```

## Trained Model and Examples

Trained model weights with VOC 0712 trainval data [ssd300_model_weights.h5](https://drive.google.com/open?id=1w_gq3WeqIveAyj4TD_09Oy6R5SJVt_hI) (~105 MB)

| Trained with VOC 0712 trainval data |
| -------------- |
| ![Man with a motorbike](http://furkanomerustaoglu.com/wp-content/uploads/2020/04/man_motorbike.png) |
| Photo by Harley-Davidson on Unsplash |
| ![Man with a dog](http://furkanomerustaoglu.com/wp-content/uploads/2020/04/man_dog_cars.png) |
| Photo by Sebastian Herrmann on Unsplash |

### References

* SSD: Single Shot MultiBox Detector [[paper]](https://arxiv.org/abs/1512.02325)
* Original caffe implementation [[code]](https://github.com/weiliu89/caffe/tree/ssd)
* ParseNet: Looking Wider to See Better [[paper]](https://arxiv.org/abs/1506.04579)
* AutoAugment: Learning Augmentation Policies from Data [[paper]](https://arxiv.org/abs/1805.09501)
* Data Augmentation Steps for SSD [[blog]](http://www.telesens.co/2018/06/28/data-augmentation-in-ssd/#Data_Augmentation_Steps)
