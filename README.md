# Neural Pipeline for Zero-Shot Data-to-Text Generation

Zero-shot data-to-text generation from RDF triples using a pipeline of pretrained language models (BART, RoBERTa). 

This repository contains code, data, and system outputs for our paper published in ACL 2022: 
> Zdeněk Kasner & Ondřej Dušek: Neural Pipeline for Zero-Shot Data-to-Text Generation. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022).

*TODO: add arXiv / anthology link*

## Model Overview
The pipeline transforms facts in natural language generated with simple single-attribute templates.

<p align="left">
  <img src="img/model.png" width=400px />
</p>

The text is generated using a three-step pipeline:
1) ordering
2) aggregation
3) paragraph compression

## Requirements
The pipeline is built using Python 3, PyTorch Lightning 1.2.5 and HuggingFace Transformers 4.12. 

See `requirements.txt` for the full list of requirements.

Installing the requirements:
```
pip install -r requirements.txt
```

## Quickstart



### System Outputs
You can find the generated descriptions from our pipeline in the [system_outputs](system_outputs) directory.


### Pretrained Models
You can download the pretrained models for individual pipeline steps here:

| model | full  |  filtered | 
|---|---|---|
| ord  |  TBA |  - |   
| agg  |  TBA |  - |  
| pc   |  TBA | TBA  |  
| pc-agg |  TBA | TBA  |  
| pc-ord-agg | TBA   | TBA  | 



### Interactive Mode
Tip: you can use any checkpoint using an interactive mode (with manual input from the command line). The input sentences are split automatically using `nltk.sent_tokenize()`.

Examples:
- **ordering:**  `./interact.py --experiment ord`
```
[In]: Blue Spice is near Burger King. Blue Spice has average customer rating. Blue Spice is a coffee shop. 
[Out]:
['Blue Spice is a coffee shop.',
 'Blue Spice is near Burger King.',
 'Blue Spice has average customer rating.']
```
- **aggregation:**  `./interact.py --experiment agg`
```
[In]: Blue Spice is a coffee shop. Blue Spice is near Burger King. Blue Spice has average customer rating.
[Out]:
[0, 1] # 0=fuse, 1=separate
```
- **paragraph compression:** `./interact.py --experiment pc_filtered`
```
[In]: Blue Spice is a coffee shop. Blue Spice is near Burger King. <sep> Blue Spice has average customer rating.
[Out]:
['Blue Spice is a coffee shop near Burger King. It has average customer '
 'rating.']
```

## WikiFluent Corpus
See the `wikifluent` directory for instructions on building the WikiFluent corpus. 

You can also **[download the generated corpus](https://owncloud.cesnet.cz/index.php/s/2oVc8UeWq4u51Od/download)** :page_facing_up:

The *filtered* version of the dataset contains examples without omissions or hallucinations (see the paper for details).

## Preprocessing

1.  Download the [WikiFluent corpus](https://owncloud.cesnet.cz/index.php/s/2oVc8UeWq4u51Od/download) and unpack it in the `data` directory.
2. Download the D2T datasets and E2E metrics:
```
./download_datasets_and_metrics.sh
```
3. Preprocess the D2T datasets. This step will parse the raw D2T datasets to prepare the data for evaluation (the data is not needed for training).
- WebNLG
```
./preprocess.py \
    --dataset webnlg \
    --dataset_dir data/d2t/webnlg/data/v1.4/en \
    --templates templates/templates-webnlg.json \
    --output data/webnlg_1stage  \
    --output_refs data/ref/webnlg  \
    --shuffle \
    --keep_separate_sents
```
- E2E
```
./preprocess.py \
    --dataset e2e \
    --dataset_dir data/d2t/e2e/cleaned-data/ \
    --templates templates/templates-e2e.json \
    --output data/e2e_1stage  \
    --output_refs data/ref/e2e  \
    --shuffle \
    --keep_separate_sents
```

## Training

### Ordering
The ordering model is trained on deshuffling the sentences from the wikifluent-*full* corpus. The model is needed for the 2-stage and 3-stage versions of the pipeline.

The implementation of the ordering model is based on https://github.com/airKlizz/passage-ordering. 
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
The aggregation model is trained on aggregating the (ordered) sentences from the wikifluent-*full* corpus.  The model is needed for the 3-stage version of the pipeline.

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

### Paragraph Compression
The paragraph compression (PC) model is trained on compressing (=fusing and rephrasing) the paragraphs from the wikifluent corpus. The following options are available:
- `MODULE`: 
  - `pc` - paragraph compression (used in the **3-stage** pipeline)
  - `pc_agg` - aggregation + paragraph compression (used in the **2-stage** pipeline)
  - `pc_ord_agg` - ordering + aggregation + paragraph compression (used in the **1-stage** pipeline)
- `VERSION`:
  - `full` - full version of the wikifluent dataset
  - `filtered` - filtered version of the wikifluent dataset
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

To use the commands below, set the environment variables to one of the following:
- `DATASET_DECODE`
  - `webnlg` - WebNLG dataset
  - `e2e` - E2E dataset
- `VERSION`:
  - `full` - module trained on the full version of the wikifluent dataset
  - `filtered` - module trained on the filtered version of the wikifluent dataset

### 3-stage
The following commands will run the 3-stage pipeline:
#### Order the data
```
./order.py \
    --experiment ord \
    --in_dir data/${DATASET_DECODE}_1stage \
    --out_dir data/${DATASET_DECODE}_2stage \
    --splits test
```

#### Aggregate the data
```
./aggregate.py \
    --experiment agg \
    --in_dir data/${DATASET_DECODE}_2stage \
    --out_dir data/${DATASET_DECODE}_3stage \
    --splits test
```
#### Apply the PC model
```
./decode.py \
    --experiment "pc_${VERSION}" \
    --module pc \
    --in_dir data/${DATASET_DECODE}_3stage \
    --split test \
    --gpus 1
```

### 2-stage
The following commands will run the 2-stage pipeline:
#### Order the data
```
./order.py \
    --experiment ord \
    --in_dir data/${DATASET_DECODE}_1stage \
    --out_dir data/${DATASET_DECODE}_2stage \
    --splits test
```

#### Apply the PC+agg model
```
./decode.py \
    --experiment "pc_agg_${VERSION}" \
    --module pc_agg \
    --in_dir data/${DATASET_DECODE}_2stage \
    --split test \
    --gpus 1
```

### 1-stage
The following command will run the 1-stage pipeline:
#### Apply the PC+ord+agg model
```
./decode.py \
    --experiment "pc_ord_agg_${VERSION}" \
    --module pc_ord_agg \
    --in_dir data/${DATASET_DECODE}_1stage \
    --split test \
    --gpus 1
```

The output is always stored in the experiment directory of the pc model (default output name is `{split}.out`).

## Evaluation
### E2E Metrics
You can re-run automatic evaluation using `evaluate.py`. The script requires [E2E metrics](https://github.com/tuetschek/e2e-metrics) (cloned by `download_datasets_and_metrics.sh`). The following command evaluates the output for the test split from the 3-stage pipeline:

```
./evaluate.py \
    --hyp_file experiments/pc/test.out \
    --ref_file data/ref/${DATASET_DECODE}/test.ref \
    --use_e2e_metrics
```
You can also run faster evaluation of BLEU score faster with the [sacreBLEU](https://pypi.org/project/sacrebleu/) package by omitting the flag `--use_e2e_metrics`.

### Semantic Accuracy

You can re-run the evaluation of semantic accuracy with [RoBERTa-MNLI](https://huggingface.co/roberta-large-mnli) with the following commands:
```
./utils/compute_accuracy.py \
    --hyp_file experiments/pc_${VERSION}/test.out \
    --ref_file data/webnlg_1stage/test.json \
    --gpu
```
This command creates a space-separated file `<hyp_file>.acc`, where each line contains a number of templates,  a number of omissions, and number of hallucinations in the respective example. 

A summary is printed using the following command:
```
./read_acc.py experiments/pc_${VERSION}/test.out.acc
```

- `or` = omission rate (omissions/facts)
- `hr` = hallucination rate (hallucinations/examples)
- `oer` = omission error rate (omissions/examples, not used in the paper)


### Ordering
*Note: the following examples are using WebNLG, computing the metrics for E2E / WikiFluent is analogous.*

For calculating the ordering metrics, first extract the reference order:
```
./preprocess.py \
    --dataset webnlg 
    --dataset_dir data/d2t/webnlg/data/v1.4/en/\
    --output data/ref/webnlg/ \
    --extract_order \
    --split test
```

Then re-run the ordering experiment with the flag `--indices_only`:
```
./order.py \
    --experiment ord \
    --in_dir data/webnlg_1stage \
    --out_dir data/webnlg_1stage_ord \
    --splits test \
    --indices_only
```

Finally, utilize the script `compute_order_metrics.py` to compute BLEU-2 and accuracy:
```
./utils/compute_order_metrics.py \
    --hyp_file data/webnlg_1stage_ord/test.out  \
    --ref_file data/ref/webnlg/test.order.out 
```

### Aggregation