NLP Repo:    

This repository provides Natural Language Models and different text classification projects.   

## Overview of Language models:      
-- paCy and NLTK (Natural Language Toolkit) for sentence and word tokenization. SpaCy is faster and better for production, while NLTK is great for research and education.  
-- SpaCy Pipeline Components: SpaCy’s pipeline consists of various attributes: tagger (`.pos_`), parser, lemmatizer (`.lemma_`), named entity recognizer (`.ent`), and dependency parser (`.dep_`). [Read more here](https://spacy.io/usage/processing-pipelines#pipelines).  
--Stemming vs. Lemmatization: Stemming uses simple rules to derive the root form of a word, while lemmatization relies on language knowledge for more accurate base forms. SpaCy does not include stemming; NLTK has both stemming and lemmatization.  
-- POS vs. Tag: Part-of-speech (POS) tagging identifies a word's coarse-grained category, while the tag provides a fine-grained category.  
-- Named Entity Recognition (NER): Identifies entities such as Person, Company, Product, Location, and Money, with options to customize entities.  
-- Stop Words: These are words that carry little meaningful information and are often ignored in text processing tasks, such as articles, prepositions, pronouns, and conjunctions.  


## Overview on approaches to covert text to vector:   
**One hot encoding, Bag of words, TF-IDF, word Embedding**   
-- One hot encoding and label encoding suffer from out of vocabulary (OOV) problem and not having fixed length representaion  
-- Bag of Words (BOW) and CountVectorizer is used to represent text data into a vector of word counts or frequencies. BOW and Bag of n-gram suffer OOV problem  
-- TF-IDF method: Term Frequency(TF) * Inverse Document Frequency(IDF) in which Term Frequency(TF) = [number of times word appeared / total no of words in a document]  
and Inverse Document Frequency(IDF) = [log(Total number of documents / number of documents that contains the word)]    
-- BOW and TF-IDF has been used for sentiment analysis on spam email using Naive Bayes  
-- Spam email data set was taken from: https://www.kaggle.com/datasets/mfaisalqureshi/spam-email?resource=download  


## Overview of Word embedding techniques:    
-- Word embedding based on CBOW and skip gram in which ANN trained on the text and a side result you get weights as word embeding vector. It includes: Word2vec, GloVe, fastText methods  
-- Word embedding based on transformer architecture: BERT, GPT  
-- Word embeddingb ased on LSTM: ELMo  
**Word2vec in Gensim and SpaCy** 
-- Gensim is a Python library and includes various models for Word2vec embeddings for generating word embeddings. It uses CBOW and skip gram to generate vector representations of words.  
-- api.info() is showing all avialible gensim models are listed here: https://github.com/piskvorky/gensim-data  
-- Gensim API Models: word2vec-google-news  
-- GloVe (Global Vectors for Word Representation) model from standford University https://nlp.stanford.edu/projects/glove/  
-- Word2Vec embedding with SpaCy lang Model  
-- W2V in gensim,GloVe and SpaCy have been used for News classification (fake detection). News data was taken from: https://www.kaggle.com/code/sharanya02/fake-news-detection/input     

## Overview of Word embedding in FastText:     
-- FastText: trained on character n-GRAM, handel OOV nicely, train for your domain (custom)    
-- FastText is both technique (method) and a Library     
-- FastText is facebook library and trained on Common Crawl and Wikipedia texts: https://fasttext.cc     
-- FastText has be used for both unsupervised and supervised text training     
-- dataset for unsupervised text training was taken from kaggle: https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset  
-- dataset for supervised text training and Ecommerce text classification was taken from: https://zenodo.org/records/3355823  


## Overview of text embedding in BERT:   
-- BERT model which is transformer based lang model  
-- BERT model was downloaded from TensorFlow hub: bert_preprocess and bert_model  
-- GPT model for Text Embedding    
-- BERT model has been used for sentiment analysis on spam email using Naive Bayes    
-- Spam email data set was taken from: https://www.kaggle.com/datasets/mfaisalqureshi/spam-email?resource=download    

## Overview of Regular Expression in NLP tasks:  
-- Regular Expression  has been used for information extraction of different texts. Take a look on this page: https://regex101.com/  



