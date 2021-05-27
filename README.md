# Knowledge-Inheritance

Source code for paper: Knowledge Inheritance for Pre-trained Language Models. The trained model parameters (in [Fairseq](https://github.com/pytorch/fairseq) format) can be downloaded from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/aab1777a161545038c01/). You can use convert_fairseq_to_huggingface.py to convert the Fairseq format into Huggingface's [transformers](https://github.com/huggingface/transformers) format easily.

## 1. Available Pretrained Models (WB: Wikipedia + BookCorpus; CS: computer science papers; BIO: biomedical papers)

### Models trained by self-learning

```
RoBERTa_WB_H_4
RoBERTa_WB_H_6
RoBERTa_WB_H_8
RoBERTa_WB_H_10
RoBERTa_WB_D_288
RoBERTa_WB_D_384
RoBERTa_WB_D_480
RoBERTa_WB_D_576
RoBERTa_WB_D_672
RoBERTa_WB_BASE
RoBERTa_WB_MEDIUM
RoBERTa_WB_BASE_PLUS
RoBERTa_WB_LARGE
GPT_WB_MEDIUM
GPT_WB_BASE
GPT_WB_BASE_PLUS
RoBERTa_CS_MEDIUM
RoBERTa_CS_BASE
RoBERTa_BIO_MEDIUM
RoBERTa_BIO_BASE
```

### Models trained by self-learning

```
RoBERTa_WB_BASE -> RoBERTa_WB_BASE_PLUS
RoBERTa_WB_BASE -> RoBERTa_WB_LARGE
RoBERTa_WB_BASE_PLUS -> RoBERTa_WB_LARGE
RoBERTa_WB_BASE -> RoBERTa_WB_BASE_PLUS -> RoBERTa_WB_LARGE
```

