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

**Warning: The instructions and the code are not complete yet, finalizing in progress.**

## Requirements
The pipeline is built using Python 3, PyTorch Lightning 1.2.5 and HuggingFace Transformers 4.12. 

See `requirements.txt` for the full list of requirements.

## WikiFluent Corpus
See the `wikifluent` directory for instructions on building the WikiFluent corpus. Alternatively, you can download the generated corpus:

- https://ufile.io/swf75rip

The *filtered* version of the dataset contains only examples which contain no omissions or hallucinations according to `roberta-mnli`.


## Preprocessing


```
./download_datasets.sh
```

```
# WebNLG
./preprocess.py 
    --dataset webnlg \
    --dataset_dir data/d2t/webnlg/data/v1.4/en \
    --templates templates/templates-webnlg.json \
    --output data/webnlg_1stage  \
    --shuffle \
    --keep_separate_sents

# E2E
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
*Integration of the ordering module is in progress. See https://github.com/airKlizz/passage-ordering for the original code.*

### Aggregation
```
./train.py \
    --dataset data/wikifluent_full \
    --experiment agg \
    --module agg \
    --gpus 1 \
    --model_name roberta-large \
    --accumulate_grad_batches 4 \
    --max_epochs 1 \
    --val_check_interval 0.05
```

### PC
```
# select dataset version: "wikifluent_filtered" "wikifluent_full"
VERSION="filtered"

# select module version: "pc" "pc_agg" "pc_ord_agg"
MODULE="pc"

./train.py \
    --dataset "wikifluent_${VERSION}" \
    --experiment "${MODULE}_${VERSION}" \
    --module "$MODULE" \
    --model_name "facebook/bart-base" \
    --max_epochs 1 \
    --accumulate_grad_batches 4 \
    --gpus 1 \
    --val_check_interval 0.05
```


## Decoding

### 3-stage

#### Order data
TODO integrate

#### Aggregate data
#### Apply PC

### 2-stage
#### Order data
TODO integrate

#### Apply PC+agg

### 1-stage
#### Apply PC+ord+agg
