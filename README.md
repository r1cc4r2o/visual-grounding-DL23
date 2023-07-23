# Visual-Grounding-DL23 
> Visual Grounding pipelines for the Deep Learning project. University of Trento, 2023
> 
> Authors: [Andrea Coppari](https://it.linkedin.com/in/andreacoppari1005), [Riccardo Tedoldi](https://www.instagram.com/riccardotedoldi/)
> 
> Supervisors: [Alessandro Conti](https://webapps.unitn.it/du/it/Persona/PER0191439/Pubblicazioni), [Elisa Ricci](https://webapps.unitn.it/du/it/Persona/PER0126701/Pubblicazioni)
> 
> [[slide-pdf]](https://drive.google.com/file/d/1xB_I5_5zepOkj-Qi5OU6e-Ak_KnJ4LIM/view?usp=sharing)
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1WgQcmtqpKZwsSXBKMNhr2V72aL6N4jua/view?usp=share_link)

Here you can check out the plug and play solution of the notebook. This notebook contains step-by-step illustation and explanation of the proposed architecture. You can run the code cells without any modifications and see the results.
## Task

Visual grounding is the process of associating linguistic information with visual content, such as images or video. It is an important task in natural language processing and computer vision, as it enables machines to understand and interpret the world in a more human-like way.

## Introduction 

The task at hand is to develop a model capable of performing visual grounding, which involves associating linguistic descriptions with visual content such as images or video. Visual grounding is a crucial task in the fields of natural language processing and computer vision, as it allows machines to better understand and interpret the world around them. This project aims to explore visual grounding by building and training a deep learning model using a dataset of image-caption pairs from RefCOCOg.

This project aims to explore visual grounding by developing a model that can accurately associate natural language descriptions with corresponding images. The model will be trained using a dataset of image-caption pairs, and evaluated on its ability to generate accurate and relevant captions for new images.


Our code and results will be shared publicly on this Git repository, so that others can reproduce our work and build upon it.


## Dataset

Our data preparation process involves applying the RefCOCOg dataset to various versions of the yolo backbone detector. This allows us to obtain the bounding box coordinates for each image region of interest. Next, we feed the cropped image regions to the CLIP-ViT-32/B model, which encodes them into high-dimensional feature vectors. We provide an extended version of the dataset that contains multiple captions per image. We used all the image-caption pairs available, instead of selecting one single caption for each image. This increased the size of the training set from 43000 to 77000 samples. The resulting data are available for download from our [gDrive folder](https://drive.google.com/drive/folders/1NPqrloMrYAlRIPGMeK2HD7i4MWgLrSK-?usp=share_link). All the preprocessed data are in the file `eval_yolo_clip_preprocessed.zip`. The data are stored in a dictionary with the following structure.

```python
import pickle

with open('data.p', 'rb') as f:
    data = pickle.load(f)
    
data
```

```bash

{
      0: {
            'image_emb': tensor([[-0.2585, -0.0283, -0.2629,  ...,  0.4138,  0.0712,  0.1495],
                              [-0.3599,  0.3452, -0.2159,  ...,  0.4407,  0.2822, -0.0598],
                              [-0.3035,  0.2639,  0.1043,  ...,  0.8638,  0.1595,  0.0069],
                              [-0.3613,  0.0264,  0.0220,  ...,  0.3015,  0.0246,  0.1188]],
                                                                              dtype=torch.float16),
            'text_emb': tensor([[-0.0849,  0.2125, -0.2272,  ...,  0.0878,  0.3113, -0.0627],
                              [ 0.1274,  0.2288, -0.2053,  ...,  0.3335,  0.1100,  0.0649]],
                                                                              dtype=torch.float16),
            'text_similarity': tensor([[6.1684e-03, 9.5947e-01, 9.4604e-04, 3.3356e-02],
                  [1.2976e-01, 6.2891e-01, 2.9106e-03, 2.3865e-01]],          dtype=torch.float16),
            'df_boxes':          xmin        ymin        xmax        ymax  confidence  class      name
                        0  230.244614   40.600266  370.827026  303.394409    0.938504      0    person
                        1  374.734375   68.870605  510.243530  262.660156    0.925634      0    person
                        2  244.321457  257.910980  385.389038  340.588043    0.851373     30      skis
                        3  413.905853   95.122055  485.417450  156.813690    0.535378     24  backpack
                        4  337.156067  211.265564  515.893433  270.108582    0.395291     30      skis,
            'caption': ['the man in yellow coat', 'Skiier in red pants.'],
            'bbox_target': [374.31, 65.06, 136.04, 201.94]},
      1: {

            ...
      
}

```

#### Download RefCOCOg dataset

```bash
gdown 1tkTUpbdkPqQ1JgHcVlsT0ikjcftF4P6x
unzip refcocog.tar.zip
rm refcocog.tar.zip
tar -xvf refcocog.tar
rm refcocog.tar
```

#### Run the evaluation script with the precomputed data

```bash
gdown 1Obq6-ApHB9dzxsmJ4Wj63aJE7rj0UDHl
tar -xvf ./eval.tar eval.py
python eval.py
rm -rf eval.tar __MACOSX
```

## Experiments

We have trained and evaluated different models using different backbones. The folder is structured as follows:
```
- `src`
| - `dataset`
| - `baseline`
| - `blipCaptioning`: Contains the code for the captioning module
| - `finetuneRephrasing`: Contains the code for the rephrasing module
| - `graphContextModule`: Contains the code for the graph context module
| - `experiments`: All the experiments are contained within the folder, 
            and they are organized into subfolders, with each notebook 
            being assigned a number and a name
```



## Results

The plot below shows the score of different solutions based on different backbones. The X-axis is the scores and the Y-axis is the evaluation across different solutions. The visualization aims to compare the score difference and to check if some detectors perform better than others. If the score is affected by the variance, is highlighted by the shadow between the line. This measure  how much the evaluations deviate from the expected value. Our proposed solutions enhance performance, as shown by the results.


![Imgur](https://i.imgur.com/Zcm51OZ.png)


![Imgur](https://i.imgur.com/XNcViSe.png)


![Imgur](https://i.imgur.com/XeFGSSg.png)


---
Legend:
+ IoU m: Mean intersection over union
+ acc: Accuracy
+ ap: Average Precision
+ ar: Average Recall
+ f1: F1 score


Here the most important model used as backbone of our experiments:

```bixtex
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, 
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}
``` 

```bixtex
@misc{li2023blip2,
      title={BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models}, 
      author={Junnan Li and Dongxu Li and Silvio Savarese and Steven Hoi},
      year={2023},
      eprint={2301.12597},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bixtex
@misc{li2022lavis,
      title={LAVIS: A Library for Language-Vision Intelligence}, 
      author={Dongxu Li and Junnan Li and Hung Le and Guangsen Wang and Silvio Savarese and Steven C. H. Hoi},
      year={2022},
      eprint={2209.09019},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bixtex
@misc{radford2021learning,
      title={Learning Transferable Visual Models From Natural Language Supervision}, 
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## To Cite:

```bibtex
@misc{CoppariTedoldi2023,
    title   = {Visual-Grounding-DL23: Visual Grounding for the Deep Learning course 2022/2023},
    author  = {Andrea Coppari, Riccardo Tedoldi},
    year    = {2023},
    url  = {https://github.com/r1cc4r2o/visual-grounding-DL23}
}
```