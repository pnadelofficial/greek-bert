# greek-bert

This repository contains code written for Beyond Translation: Opening up the Human Record, a Schmidt Sciences Humanities and AI Virtual Institute (HAVI) grant. This project seeks to develop a cutting-edge Ancient Greek Bi-directional Encoder Representations from Transformers (BERT) model, as well as a set of associated downstream tools or models built from it. 

## Parts
This repository is dynamic and growing. It is composed of several sections enumerated below:
* *scripts*: contains the BERT training script, as well as SBATCH .sh scripts for running jobs on HPC Clusters. You can find the completed BERT model here: https://huggingface.co/pnadel/ancient-greek-bert.
* *lemma-pos*: contains code for training and evaluating a morphological tagger for Ancient Greek, which outputs lemmata and morphological tags given a surface form. You can find the completed tagger here: https://huggingface.co/pnadel/ancient-greek-morph-tagger.
* *wsd*: contains code for training and evaluating a word sense disambiguator, as a way to quickly evaulate BERT models on a complex downstream task. A completed model has not yet been pushed to HuggingFace but is available upon request.
* *spacy_code* contains code for training and evaluating a `spaCy` syntactic parser for Ancient Greek. This model attains the best UAS and LAS scores for any Ancient Greek syntactic parser. Evaluation metrics and a complete model can be found here: https://huggingface.co/pnadel/ancient-greek-parser.
* *sbert* contains code for training and evaluating a `SentenceTranformer` SBERT model. This is not yet complete.


