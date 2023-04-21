# LUPerson
Unsupervised Pre-training for Person Re-identification (LUPerson).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-pre-training-for-person-re/person-re-identification-on-msmt17)](https://paperswithcode.com/sota/person-re-identification-on-msmt17?p=unsupervised-pre-training-for-person-re)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-pre-training-for-person-re/person-re-identification-on-dukemtmc-reid)](https://paperswithcode.com/sota/person-re-identification-on-dukemtmc-reid?p=unsupervised-pre-training-for-person-re)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-pre-training-for-person-re/person-re-identification-on-market-1501)](https://paperswithcode.com/sota/person-re-identification-on-market-1501?p=unsupervised-pre-training-for-person-re)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-pre-training-for-person-re/person-re-identification-on-cuhk03-labeled)](https://paperswithcode.com/sota/person-re-identification-on-cuhk03-labeled?p=unsupervised-pre-training-for-person-re)

The repository is for our CVPR2021 paper [Unsupervised Pre-training for Person Re-identification](https://arxiv.org/abs/2012.03753).

## LUPerson Dataset
LUPerson is currently the largest unlabeled dataset for Person Re-identification, which is used for Unsupervised Pre-training. LUPerson consists of 4M images of over 200K identities and covers a much diverse range of capturing environments.

**LUPerson can only be used for research, commercial usage is forbidden.**

**Details can be found at ./LUP**.

## Pre-trained Models
| Model | path |
| :------: | :------: |
| ResNet50 | [R50](https://drive.google.com/file/d/1pFyAdt9BOZCtzaLiE-W3CsX_kgWABKK6/view?usp=sharing) |
| ResNet101 | [R101](https://drive.google.com/file/d/1Ckn0iVtx-IhGQackRECoMR7IVVr4FC5h/view?usp=sharing) |
| ResNet152 | [R152](https://drive.google.com/file/d/1nGGatER6--ZTHdcTryhWEqKRKYU-Mrl_/view?usp=sharing) |

## Finetuned Results
For MGN with ResNet50:

|Dataset | mAP | cmc1 | path |
|:------:|:---:|:----:|:----:|
| MSMT17 | 66.06/79.93 | 85.08/87.63 | [MSMT](https://drive.google.com/file/d/1bV27gwAsX8L3a3yhLoxAJueqrGmQTodV/view?usp=sharing) |
| DukeMTMC | 82.27/91.70 | 90.35/92.82 | [Duke](https://drive.google.com/file/d/1leUezGnwFu8LKG2N8Ifd2Ii9utlJU5g4/view?usp=sharing) |
| Market1501 | 91.12/96.16 | 96.26/97.12 | [Market](https://drive.google.com/file/d/1AlXgY5bI0Lj7HClfNsl3RR8uPi2nq6Zn/view?usp=sharing) |
| CUHK03-L | 74.54/85.84 | 74.64/82.86 | [CUHK03](https://drive.google.com/file/d/1BQ-zeEgZPud77OtliM9md8Z2lTz11HNh/view?usp=sharing)|

These numbers are a little different from those reported in our paper, and most are slightly better.

For MGN with ResNet101:
|Dataset | mAP | cmc1 | path |
|:------:|:---:|:----:|:----:|
| MSMT17 | 68.41/81.12 | 86.28/88.27 | - |
| DukeMTMC | 84.15/92.77 | 91.88/93.99 | - |
| Market1501 | 91.86/96.21 | 96.56/97.03 | - |
| CUHK03-L | 75.98/86.73 | 75.86/84.07 | - |

**The numbers are in the format of `without RR`/`with RR`**.


## Citation
If you find this code useful for your research, please cite our paper.
```
@article{fu2020unsupervised,
  title={Unsupervised Pre-training for Person Re-identification},
  author={Fu, Dengpan and Chen, Dongdong and Bao, Jianmin and Yang, Hao and Yuan, Lu and Zhang, Lei and Li, Houqiang and Chen, Dong},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2021}
}
```

## News
We extend our `LUPerson` to `LUPerson-NL` with `Noisy Labels` which are generated from tracking algorithm, Please check for our CVPR22 paper [Large-Scale Pre-training for Person Re-identification with Noisy Labels](https://arxiv.org/abs/2203.16533). And LUPerson-NL dataset is available at https://github.com/DengpanFu/LUPerson-NL


## Third-party Usage
`LUPerson` and `LUPerson-NL` are used by some work and have obtained very good performance.
* [2021-03] [TransReID](https://openaccess.thecvf.com/content/ICCV2021/papers/He_TransReID_Transformer-Based_Object_Re-Identification_ICCV_2021_paper.pdf): https://github.com/damo-cv/TransReID
* [2021-12] [TransReID-SSL](https://arxiv.org/pdf/2111.12084.pdf): https://github.com/damo-cv/TransReID-SSL
* [2023-03] [SOLIDER](https://arxiv.org/abs/2303.17602): https://github.com/tinyvision/SOLIDER
