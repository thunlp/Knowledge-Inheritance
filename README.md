# Knowledge-Inheritance

Source code for our NAACL 2022 paper: Knowledge Inheritance for Pre-trained Language Models.

The trained model parameters (in [Fairseq](https://github.com/pytorch/fairseq) format) can be downloaded from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/aab1777a161545038c01/). Please follow [ELLE](https://github.com/thunlp/ELLE) to convert the trained checkpoint from Fairseq format into Huggingface [transformers](https://github.com/huggingface/transformers) format.

We also provide the pre-training data (already processed in fairseq format) we use in [google drive](https://drive.google.com/drive/folders/1l1cuN9JQUqZTM_1NFNtetfiXMKWqGTUo?usp=sharing), covering five pre-training domains (WB, News, Reviews, BIO and CS). We sample around 3400M tokens for each domain.

We refer the downstream performance evaluation to the implementation of [Fairseq](https://github.com/pytorch/fairseq) (GLUE tasks) and [Don't Stop Pre-training](https://github.com/allenai/dont-stop-pretraining) (ACL-ARC / CHEMPROT). For ACL-ARC / CHEMPROT, please refer to [ELLE](https://github.com/thunlp/ELLE) for easy implementation.

If you have any question, feel free to contact me by email (yujiaqin16@gmail.com).

## Installation

``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

## Pre-training under KI

``` bash
cd examples/roberta
bash train_base_to_base_plus.sh
```

For pre-training corpus collection, since BookCorpus is not publically available, you need to crawl it by yourself. We refer to [StackingBERT](https://github.com/gonglinyuan/StackingBERT) for data collection.

## Downstream evaluation

For downstream evaluation, (1) GLUE: we refer to the implementation of [Fairseq](https://github.com/pytorch/fairseq); (2) ACL-ARC & CHEMPROT: first use convert_fairseq_to_huggingface.py to convert the Fairseq format into Huggingface's [transformers](https://github.com/huggingface/transformers) format, then test the performance using the implementation of [Don't Stop Pre-training](https://github.com/allenai/dont-stop-pretraining).
