# Neural Pipeline for Zero-Shot Data-to-Text Generation

This repository contains code for zero-shot data-to-text generation from RDF triples using a pipeline of pretrained language models. 

The paper is currently under anonymous review in ACL Rolling Review (November 2021): https://openreview.net/forum?id=Lz2fD-uQyeh

## Model Overview
The model operates over the facts in natural language generated with simple single-attribute templates.

<p align="left">
  <img src="img/model.png" width=400px />
</p>

The text is generated using a three-step pipeline:
1) fact ordering
2) fact aggregation
3) paragraph compression

<!--## Quickstart
TODO
 ### Pipeline
1. Install the requirements:
```
pip install -r requirements.txt
```
2. Download the datasets:
```
./download_datasets.sh
```
3. Download the pretrained models:
- fact ordering
- fact aggregation
- paragraph compression
4. Run the pipeline:
```
./run_pipeline.sh --dataset webnlg --gpus 1
```
### Interactive Mode
You can also use any of the models in interactive mode (with manual input): see `interact.py`. -->

**Note: The instructions and the code are being improved. We will finalize the repository after the anonymization period.**

## Requirements
The pipeline is built using Python 3, PyTorch Lightning 1.2.5 and HuggingFace Transformers 4.12. 

See `requirements.txt` for the full list of requirements.


## WikiFluent Corpus
See the `wikifluent` directory for instructions on building the WikiFluent corpus. Alternatively, you can download the generated corpus:

- https://ufile.io/swf75rip

*Note: the link is temporary for the anonymization period. We will add permanent links as soon as possible.*

The *filtered* version of the dataset contains only examples which contain no omissions or hallucinations according to `roberta-mnli`.

## Pretrained Models
You can download the pretrained models for individual pipeline steps here:

- **ordering**: https://ufile.io/hrz5yq8l  (see https://github.com/airKlizz/passage-ordering)
- **aggregation**: https://ufile.io/h9mp3wzo
- **paragraph compression (pc-filtered)**: https://ufile.io/4uvkbup5

*Note: the links are temporary for the anonymization period. We will add permanent links as soon as possible. We will also upload the full variety of the pretrained models, including the non-filtered, 2-stage and 1-stage models.*


## Preprocessing

1.  Download the [WikiFluent corpus](https://ufile.io/swf75rip) and place it in the `data` directory.
2. Download the D2T datasets:
```
./download_datasets.sh
```
3. Preprocess the D2T datasets:
- WebNLG
```
./preprocess.py 
    --dataset webnlg \
    --dataset_dir data/d2t/webnlg/data/v1.4/en \
    --templates templates/templates-webnlg.json \
    --output data/webnlg_1stage  \
    --shuffle \
    --keep_separate_sents
```
- E2E
```
./preprocess.py 
    --dataset e2e \
    --dataset_dir data/d2t/e2e/cleaned-data/ \
    --templates templates/templates-e2e.json \
    --output data/e2e_1stage  \
    --shuffle \
    --keep_separate_sents
```

## Training

### Ordering
```
./train.py \
    --in_dir "data/wikifluent_full" \
    --experiment ord \
    --module ord \
    --gpus 1 \
    --model_name facebook/bart-base \
    --accumulate_grad_batches 4 \
    --max_epochs 1 \
    --val_check_interval 0.05
```

### Aggregation
```
./train.py \
    --in_dir "data/wikifluent_full" \
    --experiment agg \
    --module agg \
    --gpus 1 \
    --model_name roberta-large \
    --accumulate_grad_batches 4 \
    --max_epochs 1 \
    --val_check_interval 0.05
```

### PC
- `MODULE`: `pc` `pc_agg` `pc_ord_agg`
- `VERSION`: `filtered` `full`
```
VERSION="filtered"
MODULE="pc"
./train.py \
    --in_dir "data/wikifluent_${VERSION}" \
    --experiment "${MODULE}_${VERSION}" \
    --module "$MODULE" \
    --model_name "facebook/bart-base" \
    --max_epochs 1 \
    --accumulate_grad_batches 4 \
    --gpus 1 \
    --val_check_interval 0.05
```


## Decoding
There are 3 possible pipelines for generating the text from data: 3-stage, 2-stage, or 1-stage (see the paper for detailed description).

```
DATASET_DECODE="webnlg"
VERSION="filtered"
```


### 3-stage

#### Order data
*Integration of the ordering module is in progress. Refer to https://github.com/airKlizz/passage-ordering for the original code.* 

*In the meantime, our script `utils/order.py` can be used in the top level directory of the [repository](https://github.com/airKlizz/passage-ordering) for ordering the dataset.*

#### Aggregate data
```
./aggregate.py \
    --experiment agg \
    --in_dir data/${DATASET_DECODE}_2stage \
    --out_dir data/${DATASET_DECODE}_3stage \
    --splits test
```
#### Apply PC model
```
./decode.py \
    --experiment "pc_${VERSION}" \
    --module pc \
    --in_dir data/${DATASET_DECODE}_3stage \
    --split test \
    --gpus 1
```

### 2-stage
#### Order data
*Integration of the ordering module is in progress. Refer to https://github.com/airKlizz/passage-ordering for the original code.* 

*In the meantime, our script `utils/order.py` can be used in the top level directory of the [repository](https://github.com/airKlizz/passage-ordering) for ordering the dataset.*


#### Apply PC+agg model
```
./decode.py \
    --experiment "pc_agg_${VERSION}" \
    --module pc_agg \
    --in_dir data/${DATASET_DECODE}_2stage \
    --split test \
    --gpus 1
```

### 1-stage
#### Apply PC+ord+agg model
```
./decode.py \
    --experiment "pc_ord_agg_${VERSION}" \
    --module pc_ord_agg \
    --in_dir data/${DATASET_DECODE}_1stage \
    --split test \
    --gpus 1
```

The output is stored in the experiment directory (`test.out`).