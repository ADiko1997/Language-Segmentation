import tensorflow as tf
import keras as k
from keras.preprocessing.sequence import pad_sequences  
import numpy as np
import sklearn 
from sklearn import preprocessing
import re
from typing import Tuple, List, Dict
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import matplotlib.pyplot as plt
from collections import Counter
import statistics

#making ready the datasets to work, this file is divided in two parts
#part1->basic preprocessing functions 
#part2-> transforming datasets

#begin: part1
"""
All the fuctions included in part1 are core preprocessing functions
"""
#devide or split in spaces
def divide(path):
    """
    This function divides the text into words (splits it if there are spaces)
    """
    tf = open(path, encoding='utf8')
    splited = []
    for i in tf:
        line = i.rstrip().split()
        splited.append(line)
    return splited
#create labels for each char

def find_max_len(path):
    """
    This function finds the longest sentence on the path file
    """
    file = join_file(path)
    maxi = 0
    for i in file:
        if len(i) > maxi:
            maxi = len(i)
    print(maxi)


def labels(corpus):
    """
    This function takes as input the dataset splited into word and for each char based
    on its position, it assigns a label B I E S
    """
    BIES = []
    for i in range(len(corpus)):
        BIES_format = []
        for word in corpus[i]:
            counter = 0
            for j in word:
                if len(word) == 1:
                    BIES_format.append("S")
                elif j == word[0] and counter == 0:
                    counter+=1
                    BIES_format.append("B")
                elif counter != 0 and counter < len(word)-1:
                    counter+=1
                    BIES_format.append("I")
                else:
                    BIES_format.append("E")
        BIES.append(("").join(BIES_format))  
    return BIES 
#join the training set by removing the spaces on each line
def training_set_func(path):
    training_x = []
    with open(path,encoding='utf8') as f:
        for line in f:
            a = line = line.rstrip().split()
            training_x.append("".join(a))
    return training_x


def labels_to_numbers(data):
    """
    This function transforms the BIES labesl into numbers, an alternative to 
    this would be LabelEncoder of ScikitLearn
    """
    label_dict = {"B":0,"I":1,"E":2,"S":3}
    y_output = []
    for word in data:
        myLabel = []
        for ch  in word:
            if label_dict.get(ch) != None:
                myLabel.append(label_dict.get(ch))
                
        y_output.append(myLabel)
            
    return y_output



def compare(data1, data2):
    """
    Testing purpose function compares the len of real text and the one transformen 
    row by row
    """
    for i in range(len(data2)):
        if data1[i] != data2[i]:
            return False
    return True

def join_file(path):
    """
    Creates a data structure which contains the text row by row without spaces
    """
    joined = []
    with open(path, encoding='utf8') as file:
        for line in file:
            a = line.rstrip().split()
            joined.append("".join(a))
    return joined

def split_into_ngrams(text, n) -> List[str]:
    # english use only text = text.lower
    """
    Split the data into n_gram based on the input of n
    """
    ngrams=[]
    for j in range(len(text)):
        line = text[j]
        for i in range(len(line)):
            ngram = line[i:i+n]
            ngrams.append(ngram)
    return ngrams

def make_vocabulary_concat(unigram, bigram):
    """
    Makes the concatenated vocabulary(unigram and bigram combined)
    """
    vocab ={'UNK' :0}
    grams = unigram + bigram
    for i in grams:
        if i not in vocab:
            vocab[i] = len(vocab)
    return vocab

def make_vocabulary1(n_gram, n):
    """
    Makes the vocabularies of uni and bi grams based on the input value of n
    """
    vocab={'UNK':0}
    for i in n_gram:
        if i not in vocab:
            if n == 2 :
                vocab[i]  = len(vocab) + 1422  #5168 is the len of vocab of unigrams so the numbers does not repeat 
            else:
                vocab[i] = len(vocab)
    return vocab

#Creatin feature vectors to give as input to the embedding layer of keras
#unigram feture vector
def feature_vector_unigram(data, vocabulary):
    """
    This function creates the feature vector for unigrams by replacing each char with its corres-
    pondin value on vacabulary
    """
    feature_vector = []
    for sentence in data:
        vector = []
        for word in sentence:
            if word not in vocabulary:
                vector.append(vocabulary['UNK'])
            else:
                vector.append(vocabulary[word])
        feature_vector.append(vector)
    return feature_vector
#bigram feature vector
def feature_vector_bigrams(text, vocab):
    """
    This function creates the feature vector for unigrams by replacing each char with its corres-
    pondin value on vacabulary
    """
    feature_vector=[]
    for j in range(len(text)):
        vector = []
        line = text[j]
        for i in range(len(line)):
            bigram = line[i:i+2]
            if bigram not in vocab:
                vector.append(vocab['UNK'])
            else:
                vector.append(vocab[bigram])
        feature_vector.append(vector)
    return feature_vector
#end: part1

#begin: part2
"""
Part 2 consist of function of functions for preprocessing the data based on its role,
separated functions for predict data, testing and training data
"""
def preprocessing_basic(path):
    """
    This function is needed for every step becazue by giving the path of the file it returns
    the vocabularies needed to preprocess the data before feeding them to the model for training
    evaluation or prediction
    """
    #creating uni and bi grams
    text = join_file(path)
    unigram = split_into_ngrams(text,1)
    bigram = split_into_ngrams(text,2)

    #creating vocabularies, two separated an   d one concatenated
    concat_vocab = make_vocabulary_concat(unigram, bigram)
    vocab_uni = make_vocabulary1(unigram, 1)
    vocab_bi = make_vocabulary1(bigram, 2)
    return concat_vocab, vocab_uni, vocab_bi, text

def preprocessing_label_transformation(path):
    #transforming the training dataset into labels
    """
    By giving the path this function generates the labeld file in numbers and in BIES format
    """
    corpus = divide(path)
    x = training_set_func(path)
    y = labels(corpus)
    data_set = pd.DataFrame()
    data_set["x"] = x
    data_set['y'] = y
    y_set = data_set['y'].copy()
    labels_to_num = labels_to_numbers(y_set)
    return labels_to_num, y_set

def preprocessing_test( vocab_uni, vocab_bi,path):
    """
    This function by giving the path to it, returns the data ready to use for evaluation
    """
    corpus_dev = divide(path)
    dev_x = training_set_func(path)
    dev_y = labels(corpus_dev)
    dev_text = join_file(path)
    max_len = find_max_len(path)
    label_to_num_dev = labels_to_numbers(dev_y)
    feature_vector_uni_dev = feature_vector_unigram(dev_text, vocab_uni)
    feature_vector_bi_dev = feature_vector_bigrams(dev_text, vocab_bi)
    dev_x_uni = pad_sequences(feature_vector_uni_dev, truncating='pre', padding='post', maxlen=30)
    dev_x_bi = pad_sequences(feature_vector_bi_dev, truncating='pre', padding='post', maxlen=30)
    dev_y = pad_sequences(label_to_num_dev, truncating='pre', padding='post', maxlen=30)
    dev_y = k.utils.to_categorical(dev_y)
    return dev_x_uni, dev_x_bi, dev_y

def preprocessing_predict( vocab_uni, vocab_bi,path):
    """
    This function takes as input the path of the file for prediction and return the data
    preprocessed for prediction
    """
    corpus_pred = divide(path)
    pred_x = training_set_func(path)
    pred_y = labels(corpus_pred)
    max_len = find_max_len(path)
    pred_text = join_file(path)
    label_to_num_pred = labels_to_numbers(pred_y)
    feature_vector_uni_pred = feature_vector_unigram(pred_text, vocab_uni)
    feature_vector_bi_pred = feature_vector_bigrams(pred_text, vocab_bi)
    pred_x_uni = pad_sequences(feature_vector_uni_pred, truncating='pre', padding='post', maxlen=max_len)
    pred_x_bi = pad_sequences(feature_vector_bi_pred, truncating='pre', padding='post', maxlen=max_len)
    pred_y = pad_sequences(label_to_num_pred, truncating='pre', padding='post', maxlen=max_len)
    pred_y = k.utils.to_categorical(pred_y)
    return pred_x_uni, pred_x_bi, pred_y ,max_len

def preprocessing_training(text, vocab_uni, vocab_bi, labels_to_num):
    """
    Makes the training data ready to be feeded to the model
    """
    #creating feature vectors for uni and bi gram
    feature_vector_uni = feature_vector_unigram(text, vocab_uni)
    feature_vector_bi = feature_vector_bigrams(text, vocab_bi)  

    #padding the data (input for test and dev) and transforming labels from numbers to onehotencoded
    train_x_uni = pad_sequences(feature_vector_uni, truncating='pre', padding='post', maxlen=30)
    train_x_bi = pad_sequences(feature_vector_bi, truncating='pre', padding='post', maxlen=30)
    train_y = pad_sequences(labels_to_num, truncating='pre', padding='post', maxlen=30)
    train_y = k.utils.to_categorical(train_y)
    return  train_x_uni, train_x_bi, train_y
#end: part2

if __name__ == "__main__":
    import sys
    preprocessing(int(sys.argv[1]))