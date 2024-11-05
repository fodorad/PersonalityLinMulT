# PersonalityLinMulT
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

LinMulT is trained for Big Five personality trait estimation using the First Impressions V2 dataset and sentiment estimation using the MOSI and MOSEI datasets.
* paper: **Multimodal Sentiment and Personality Perception Under Speech: A Comparison of Transformer-based Architectures** ([pdf](https://proceedings.mlr.press/v173/fodor22a/fodor22a.pf), [website](https://proceedings.mlr.press/v173/fodor22a.html))

# Setup
### Install package from PyPI for inference
```
pip install personalitylinmult
```

### Install package for training
```
git clone https://github.com/fodorad/PersonalityLinMulT
cd PersonalityLinMulT
pip install -e .[all]
pip install -U -r requirements.txt
```

#### Supported extras definitions:
| extras tag | description |
| --- | --- |
| train | dependencies for feature extraction, training the model from scratch and visualization |
| all | extends the 'train' dependencies for development. currently it is the same as 'train' tag |


# Related projects

### exordium
Collection of preprocessing functions and deep learning methods. This repository contains revised codes for fine landmark detection (including face, eye region, iris and pupil landmarks), head pose estimation, and eye feature calculation.
* code: https://github.com/fodorad/exordium

### (2022) LinMulT
General-purpose Multimodal Transformer with Linear Complexity Attention Mechanism. This base model is further modified and trained for various tasks and datasets.
* code: https://github.com/fodorad/LinMulT

### (2023) BlinkLinMulT
LinMulT is trained for blink presence detection and eye state recognition tasks.
Our results demonstrate comparable or superior performance compared to state-of-the-art models on 2 tasks, using 7 public benchmark databases.
* paper: BlinkLinMulT: Transformer-based Eye Blink Detection ([pdf](https://adamfodor.com/pdf/2023_Fodor_Adam_MDPI_BlinkLinMulT.pdf), [website](https://www.mdpi.com/2313-433X/9/10/196))
* code: https://github.com/fodorad/BlinkLinMulT


# Citation - BibTex
If you found our research helpful or influential please consider citing:

### (2023) BlinkLinMulT for blink presence detection and eye state recognition
```
@Article{fodor2023blinklinmult,
  title = {BlinkLinMulT: Transformer-Based Eye Blink Detection},
  author = {Fodor, Ádám and Fenech, Kristian and Lőrincz, András},
  journal = {Journal of Imaging},
  volume = {9},
  year = {2023},
  number = {10},
  article-number = {196},
  url = {https://www.mdpi.com/2313-433X/9/10/196},
  PubMedID = {37888303},
  ISSN = {2313-433X},
  DOI = {10.3390/jimaging9100196}
}
```

### (2022) LinMulT for personality trait and sentiment estimation
```
@InProceedings{pmlr-v173-fodor22a,
  title = {Multimodal Sentiment and Personality Perception Under Speech: A Comparison of Transformer-based Architectures},
  author = {Fodor, {\'A}d{\'a}m and Saboundji, Rachid R. and Jacques Junior, Julio C. S. and Escalera, Sergio and Gallardo-Pujol, David and L{\H{o}}rincz, Andr{\'a}s},
  booktitle = {Understanding Social Behavior in Dyadic and Small Group Interactions},
  pages = {218--241},
  year = {2022},
  editor = {Palmero, Cristina and Jacques Junior, Julio C. S. and Clapés, Albert and Guyon, Isabelle and Tu, Wei-Wei and Moeslund, Thomas B. and Escalera, Sergio},
  volume = {173},
  series = {Proceedings of Machine Learning Research},
  month = {16 Oct},
  publisher = {PMLR},
  pdf = {https://proceedings.mlr.press/v173/fodor22a/fodor22a.pdf},
  url = {https://proceedings.mlr.press/v173/fodor22a.html}
}
```

# Contact
* Ádám Fodor (foauaai@inf.elte.hu)