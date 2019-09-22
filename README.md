# 2019B - Final Project in Introuction to NLP @OpenU
### This is the work of `@ira-vitenzon` and `@shaharlinial`

We aim to create a sequence to sequence model that can detect metaphors
(in the english language, but can be ported out to any language given a
large dataset of metaphores)

This project is made with PyTorch.


### Installation Notes:

*Automatic Installation (Ubuntu Only):* 

1. Clone this repository with `git clone`
2. Get into repo directory `cd deep-meatphor-detection`
3. Run `sudo bash installation.sh`
4. Run `python3 run.py`


*Manual Installation:*

1. Clone or Download this repository.
2. Download
   [Glove by NLP Stanford Common Crawl](http://nlp.stanford.edu/data/glove.840B.300d.zip)
3. Create a directory named `glove` inside the repository directory
4. Extract `glove.840B.300d.zip` to `glove.840B.300d.txt`
5. Rename `glove.840B.300d.txt` into `glove840B300d.txt`
6. Run `python3 run.py`

```
deep-metaphor-detection
│   README.md
│   model.py
 |    run.py    
 |    requirements.txt
 |    util.py
 |    seq2seq_with_attention_model.py
│
└───glove
│   │   glove840B300d.txt    
```