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

### References

* SSD: Single Shot MultiBox Detector [[paper]](https://arxiv.org/abs/1512.02325)
* Original caffe implementation [[code]](https://github.com/weiliu89/caffe/tree/ssd)
* ParseNet: Looking Wider to See Better [[paper]](https://arxiv.org/abs/1506.04579)
* AutoAugment: Learning Augmentation Policies from Data [[paper]](https://arxiv.org/abs/1805.09501)
* Data Augmentation Steps for SSD [[blog]](http://www.telesens.co/2018/06/28/data-augmentation-in-ssd/#Data_Augmentation_Steps)
