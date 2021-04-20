# LUPerson
Unsupervised Pre-training for Person Re-identification (LUPerson)

The repository is for our CVPR2021 paper [Unsupervised Pre-training for Person Re-identification](https://arxiv.org/abs/2012.03753).

## LUPerson Dataset
LUPerson is currently the largest unlabeled dataset for Person Re-identification, which is used for Unsupervised Pre-training. LUPerson consists of 4M images of over 200K identities and covers a much diverse range of capturing environments. 

**To Be Added**

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

<!-- <table>
    <tr>
        <td>Dataset</td>
        <td>mAP</td>
        <td>cmc1</td>
        <td>model</td>
    </tr>
    <tr>
        <td>MSMT17</td>
        <td>66.06|79.93</td>
        <td>85.08|87.63</td>
        <td>0</td>
    </tr>
    <tr>
        <td>DukeMTMC</td>
        <td>82.27|91.70</td>
        <td>90.35|92.82</td>
        <td>0</td>
    </tr>
    <tr>
        <td>Market</td>
        <td>91.12|96.16</td>
        <td>96.26|97.12</td>
        <td>0</td>
    </tr>
    <tr>
        <td>CUHK03-L</td>
        <td>74.54|85.84</td>
        <td>74.64|82.86</td>
        <td> [CUHK03](https://drive.google.com/file/d/1BQ-zeEgZPud77OtliM9md8Z2lTz11HNh/view?usp=sharing) </td>
    </tr>
</table>
 -->