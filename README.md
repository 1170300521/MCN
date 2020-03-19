# Multi-task Collaborative Network for Joint Referring Expression Comprehension and Segmentation

Multi-task Collaborative Network for Joint Referring Expression Comprehension and Segmentation

by Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Liujuan Cao, Chenglin Wu, Cheng Deng and Rongrong Ji.

IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020, Oral

## Introduction

This repository is keras implementation of MCN.  The principle of MCN is a multimodal and multitask collaborative learning framework. In MCN, RES can help REC to achieve better language-vision alignment, while REC can help RES to better locate the referent. In addition, we address a key challenge in this multi-task setup, i.e., the prediction conflict, with two innovative designs namely, Consistency Energy Maximization (CEM) and Adaptive Soft Non-Located Suppression (ASNLS).  The network structure is illustrated as following:

<p align="center">
  <img src="https://github.com/luogen1996/MCN/blob/master/fig1.png" width="90%"/>
</p>

## Citation

    @inproceedings{luo2020multi,
      title={Multi-task Collaborative  Network for Joint  Referring Expression Comprehension and Segmentation},
      author={Luo, Gen and Zhou, Yiyi and Sun, Xiaoshuai and Cao, Liujuan and Wu, Chenglin and
      Deng, Cheng and Ji Rongrong},
      booktitle={CVPR},
      year={2020}
    }
## Prerequisites

- Python 3.6

- tensorflow-1.9.0 for cuda 9 or tensorflow-1.14.0 for cuda10

- keras-2.2.4

- spacy (you should download the glove embeddings by running `spacy download en_vectors_web_lg` )

- Others (progressbar2, opencv, etc. see requirement.txt)

## Data preparation

-  Follow the instructions of  DATA_PRE_README.md to generate training data and testing data of RefCOCO, RefCOCO+ and RefCOCOg.

-  Download the pretrained weights of backbone (vgg and darknet). We provide pretrained weights of keras  version for this repo and another  darknet version for  facilitating  the researches based on pytorch or other frameworks.  All pretrained backbones are trained for 450k iterations on COCO 2014 *train+val*  set while removing the images appeared in the *val+test* sets of RefCOCO, RefCOCO+ and RefCOCOg (nearly 6500 images).  Please follow the instructions of  DATA_PRE_README.md to download them.

## Training 

1. Preparing your settings. To train a model, you should  modify ``./config/config.json``  to adjust the settings  you want. The default settings are used for RefCOCO, which are easy to achieve 80.0 and 62.0  accuracy for REC and RES respectively on the *val* set.
2. Training the model. run ` train.py`  under the main folder to start training:
```
python train.py
```
3. Testing the model.  You should modify  the setting json to check the model path ``evaluate_model`` and dataset ``evaluate_set`` using for evaluation.  Then, you can run ` test.py`  by
```
python test.py
```
​	After finishing the evaluation,  a result file will be generated  in ``./result`` folder.

4. Training log.  Logs are stored in ``./log`` directory, which records the detailed training curve and accuracy per epoch. If you want to log the visualizations, please  setting  ``log_images`` to 1 in ``config.json``.   By using tensorboard you can see the training details like below：
  <p align="center">
  <img src="https://github.com/luogen1996/MCN/blob/master/fig2.png" width="90%"/>
  </p>

## Credits

 Thanks for a lot of codes from [keras-yolo3](https://github.com/qqwweee/keras-yolo3) , [keras-retinanet](https://github.com/fizyr/keras-retinanet)  and the framework of  [darknet](https://github.com/AlexeyAB/darknet) using for backbone pretraining.
