# COALA: A Neural Coverage-Based Approach for Long Answer Selection with Small Data

This repository contains the data and code to reproduce the results of our paper: 
https://public.ukp.informatik.tu-darmstadt.de/aaai19-coala-cqa-answer-selection/2019_AAAI_COALA_Camready.pdf 

Please use the following citation:

```
@article{rueckle:AAAI:2019,
  title = {{COALA}: A Neural Coverage-Based Approach for Long Answer Selection with Small Data.},
  author = {R{\"u}ckl{\'e}, Andreas and Moosavi, Nafise Sadat and Gurevych, Iryna},
  publisher = {Association for Computational Linguistics},
  booktitle = {Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI 2019)},
  pages = {to appear},
  month = jan,
  year = {2019},
  location = {Honolulu, Hawaii, USA},
  doi = "",
  url = ""
}
```

> **Abstract:** Current neural network based community question answering (cQA) systems fall short of (1) properly 
  handling long answers which are common in cQA; (2) performing under small data conditions, where a large amount of 
  training data is unavailable—i.e., for some domains in English and even more so for a huge number of datasets in other
  languages; and (3) benefiting from syntactic information in the model—e.g., to differentiate between identical lexemes
  with different syntactic roles. In this paper, we propose COALA, an answer selection approach that (a) selects 
  appropriate long answers due to an effective comparison of all question-answer aspects, (b) has the ability to 
  generalize from a small number of training examples, and (c) makes use of the information about syntactic roles of
  words. We show that our approach outperforms existing answer selection models by a large margin on six cQA datasets
  from different domains. Furthermore, we report the best results on the passage retrieval benchmark WikiPassageQA.


Contact person: Andreas Rücklé

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


> This repository contains experimental software and is published for the sole purpose of giving additional background 
  details on the respective publication. 


## Usage

To run an experiment:
```
python run_experiment configs/se_apple_coala.yaml
```

To run hyperparameter optimization:
```
python run_random_search.py configs/example_random_search.yaml
```

The datasets are available here: [coala_data.zip](https://public.ukp.informatik.tu-darmstadt.de/aaai19-coala-cqa-answer-selection/coala_data.zip)

More details on the framework that we used in COALA can be found in our other repositories:
[iwcs2017-answer-selection](https://github.com/UKPLab/iwcs2017-answer-selection),
[acl2017-non-factoid-qa](https://github.com/UKPLab/acl2017-non-factoid-qa/tree/master/Candidate-Ranking)


## Dependencies and Requirements

We used Python 2.7.14 for our experiments. The output of ```pip --freeze``` is given in pipfreeze.txt (not all of the
packages are strictly required). 


## Running Sentence Embedding Baselines

This requires creating a new config file. Use the ```qa.train.no_training``` trainer module and the  ```qa.evaluation.evaluation_embeddings_server_sents``` evaluation module. [PMeans](https://arxiv.org/abs/1803.01400) embeddings can be evaluated with their [embedding server](https://github.com/UKPLab/arxiv2018-xling-sentence-embeddings/tree/master/model). The [embedding_types](https://github.com/UKPLab/aaai2019-coala-cqa-answer-selection/blob/master/experiment/qa/evaluation/evaluation_embeddings_server_sents.py#L54) need to be set accordingly.