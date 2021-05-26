# Fast ReID with modification for our finetuning tasks.
We modify the code from [fast-reid](https://github.com/JDAI-CV/fast-reid) with commits on 10/9/2020(nearby). Please follow [fast-reid](https://github.com/JDAI-CV/fast-reid)'s instruction to install fast-reid.

## Dataset
We reorganized the data structure, in order to do experiments with ``small-scale`` and ``few-shot`` setting.

All the four datasets: CUHK03_NP, Market1501, DukeMTMC, MSMT17 are in the same organized structure as standard Market1501.

>datasets
>>DATANAME
>>>bounding_box_train
>>>
>>>bounding_box_test
>>>
>>>query
>>>
>>>few_ids
>>>
>>>few_ims


* DATANAME can be one of ``CUHK03_NP, Market1501, DukeMTMC, MSMT17``.
* few_ids are the split for ``small-scale`` with different percentage.
* few_ims are the split for ``few-shot`` with different percentage.
These splits can be found at fast-reid-ROOT/datasets/DATANAME/\*.

For CUHK03_NP, it follows the same settings proposed in Re-Rank, and details can be found at [RR](https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP).

For MSMT17, we adopt the structure reformulated by [ZhunZhong](https://github.com/zhunzhong07/ECN).

For Market1501 and DukeMTMC, original data setting are used.

We re-implemented dataset class for ``CUHK03_NP, Market1501, DukeMTMC, MSMT17`` and refer it as ``CMDM`` in fast-reid-ROOT/fastreid/data/datasets/cmdm.py. It needs some specific arguments:

* `root`: data root.
* `data_name`: dataset name, one of ``CUHK03_NP, Market1501, DukeMTMC, MSMT17``.
* `split_mode`: `id` for `small-scale`, `im` for `few-shot`, `ori` for `don't split`.
* `split_ratio`: split ratio, one of `[0.1, 0.2, ... , 1.0]`.
* `repeat_ratio`: repeat ratio, when each person has extremely few images for `few-shot` setting, it is necessary to repeat the sampled sub-set. Usually, 2 or 3 is good enough.


## Configs
We add some new config files for training MGN with our pre-trained models at fast-reid-ROOT/configs/CMDM/\*.


## Train

```shell
GPUS=0,1,2,3
DATASET=market
SPLIT=id
RATIO=1.0
CUDA_VISIBLE_DEVICES=${GPUS} python tools/train_net.py --num-gpus 4 \
    --config-file ./configs/CMDM/mgn_R50_moco.yml \
    MODEL.BACKBONE.PRETRAIN_PATH "pre_models/LUP/lup_moco_r50.pth" \
    DATASETS.ROOT "datasets" INPUT.DO_AUTOAUG False TEST.EVAL_PERIOD 60 \
    DATASETS.KWARGS "data_name:${DATASET}+split_mode:${SPLIT}+split_ratio:${RATIO}" \
    OUTPUT_DIR "logs/lup_moco_r50/${DATASET}/${SPLIT}_${RATIO}"
```

## Test
Suppose you have downloaded our finetuned models or you have finetuned your own models, and put them at `your_finetuned_model_dir`.
```shelll
DATASET=market
PATH_TO_CHECKPOINT_FILE='your_finetuned_model_dir/market.pth'
python tools/train_net.py --eval-only \
    --config-file ./configs/CMDM/mgn_R50_moco.yml \
    DATASETS.ROOT "datasets" DATASETS.KWARGS "data_name:${DATASET} \
    MODEL.WEIGHTS ${PATH_TO_CHECKPOINT_FILE} MODEL.DEVICE "cuda:0" \
    OUTPUT_DIR "logs/lup_moco/test/${DATASET}"
```
