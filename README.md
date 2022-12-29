# Dacon_Sentiment_Type_classification

## Result

|         |   F1   | RANK |
|:-------:|:------:|:----:|
| Public  | 0.7552 |  15   |
| Private | 0.75359 |  11   |


## The main problem and how to solve it

- class 불균형

> * loss 계산 비중 조정
> * generalization
> * Oversampling

- multi label text classification vs text classification for #4

> * 2가지 모두 사용


**..to be continue**


# Reference

## Reference about loss function

- [R-Drop: Regularized Dropout for Neural Networks (NeurlPS 2021)](https://arxiv.org/pdf/2106.14448v2.pdf)
	- [Reference Github link](https://github.com/dropreg/R-Drop)

- [SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization (ACL 2020)](https://aclanthology.org/2020.acl-main.197.pdf)

- [Dual Contrastive Learning: Text Classification via Label-Aware Data Augmentation (2022)](https://arxiv.org/pdf/2201.08702v1.pdf)

- [Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.643.pdf)

## Reference about Model

- [An Algorithm for Routing Vectors in Sequences (2022)](https://arxiv.org/pdf/2211.11754.pdf)
  - Heinsen routing Algorithm
