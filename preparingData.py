#!/usr/bin/env python
# coding: utf-8

# In[6]:
import pickle
import utils
import pandas as pd  
import numpy as np
import sys
import time
import re
import os
import gc
import csv
from ast import literal_eval
import seaborn as sns
from operator import add
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve, auc
from sklearn.utils import resample
from tensorflow.keras.preprocessing.sequence import pad_sequences
plt.style.use('fivethirtyeight')
import tensorflow as tf
import keras
from keras.layers import Layer 
from tensorflow.keras.layers import InputSpec
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import Model, model_from_json, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout, Dense, Embedding, LSTM, SpatialDropout1D, Input, MaxPooling1D, Flatten, GRU, Activation, Conv2D, Reshape, MaxPool2D, concatenate, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, BatchNormalization,GlobalMaxPool1D, Add, TimeDistributed, LeakyReLU,GaussianNoise, GaussianDropout
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, Callback, EarlyStopping
use_gpu=True
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
from numpy import dstack
from keras.layers.core import Lambda
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words("english")
from nltk import FreqDist
# In[4]:

urlfil="folder"


# In[2]:


def write_status(i, total):
    ''' Writes status of a process to console '''
    sys.stdout.write('\r')
    sys.stdout.write('Processing %d/%d' % (i, total))
    sys.stdout.flush()
    
def preprocess_word(word):
    word = word.strip('\'"?!,.():;')
    word = re.sub(r'(.)\1+', r'\1\1', word)
    word = re.sub(r'(-|\')', '', word)
    return word

# In[3]:

def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


# In[5]:

def handle_emojis(abstract):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    abstract = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', abstract)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    abstract = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', abstract)
    # Love -- <3, :*
    abstract = re.sub(r'(<3|:\*)', ' EMO_POS ', abstract)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    abstract = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', abstract)
    # Sad -- :-(, : (, :(, ):, )-:
    abstract = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', abstract)
    # Cry -- :,(, :'(, :"(
    abstract = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', abstract)
    return abstract


# In[6]:


def preprocess_abstract(abstract):
    processed_abstract = []
    # Convert to lower case
    abstract = abstract.lower().replace('\r\n', '')
    # Replaces URLs with the word URL
    abstract = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', abstract)
    # Replace @handle with the word USER_MENTION
    abstract = re.sub(r'@[\S]+', 'USER_MENTION', abstract)
    # Replaces #hashtag with hashtag
    abstract = re.sub(r'#(\S+)', r' \1 ', abstract)
    # Remove RT (abstract)
    abstract = re.sub(r'\brt\b', '', abstract)
    # Replace 2+ dots with space
    abstract = re.sub(r'\.{2,}', ' ', abstract)
    # Strip space, " and ' from abstract
    abstract = abstract.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    abstract = handle_emojis(abstract )
    # Replace multiple spaces with a single space
    abstract = re.sub(r'\s+', ' ', abstract)
    words = abstract.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            if use_stemmer:
                word = str(porter_stemmer.stem(word))
            processed_abstract.append(word)

    return ' '.join(processed_abstract)


# In[7]:

def preprocess_csv(csv_file_name, processed_file_name, test_file=False):
    save_to_file = open(processed_file_name, 'w')
    with open(csv_file_name, 'r', encoding = "utf-8") as csv:
        lines = [line.strip() for line in csv.readlines() if line.strip()]
        total = len(lines)
        print(f'total: {total}')
        sameCount = 0
        sameLines = []
        for i, line in enumerate(lines):
            abstract = line[:line.find(',')].strip()
            category = line[1 + line.find(','):].strip()     
            processed_abstract = preprocess_abstract(abstract)
            if processed_abstract in sameLines:            
                sameLines.append(processed_abstract)
                sameCount+=1
            else:
                save_to_file.write(f'{processed_abstract},{category}\n')
    print(f'same: {sameCount}')   
    save_to_file.close()
    return processed_file_name

# In[8]:

if len(sys.argv) != 2:
    print(sys.argv[0])
use_stemmer = False
csv_file_name = urlfil+ "data/dataset.csv"
processed_file_name = urlfil+ "data/processed_datasent.csv"
print(processed_file_name)
preprocess_csv(csv_file_name, processed_file_name, test_file=False)

# In[9]:

def analyze_abstract(abstract):
    result = {}
    result['MENTIONS'] = abstract.count('USER_MENTION')
    result['URLS'] = abstract.count('URL')
    result['POS_EMOS'] = abstract.count('EMO_POS')
    result['NEG_EMOS'] = abstract.count('EMO_NEG')
    abstract = abstract.replace('USER_MENTION', '').replace('URL', '')
    words = abstract.split()
    result['WORDS'] = len(words)
    bigrams = get_bigrams(words)
    result['BIGRAMS'] = len(bigrams)
    return result, words, bigrams

# In[10]:

def get_bigrams(abstract_words):
    bigrams = []
    num_words = len(abstract_words)
    for i in range(num_words - 1):
        bigrams.append((abstract_words[i], abstract_words[i + 1]))
    return bigrams

# In[11]:

def get_bigram_freqdist(bigrams):
    freq_dict = {}
    for bigram in bigrams:
        if freq_dict.get(bigram):
            freq_dict[bigram] += 1
        else:
            freq_dict[bigram] = 1
    counter = Counter(freq_dict)
    return counter

# In[13]:

num_abstract, num_human, num_paraphrased, num_gpt = 0, 0, 0, 0
num_mentions, max_mentions = 0, 0
num_emojis, num_pos_emojis, num_neg_emojis, max_emojis = 0, 0, 0, 0
num_urls, max_urls = 0, 0
num_words, num_unique_words, min_words, max_words = 0, 0, 1e6, 0
num_bigrams, num_unique_bigrams = 0, 0
all_words = []
all_bigrams = []
with open(urlfil+'data/processed_dataset.csv', 'r') as csv:
    lines = csv.readlines()
    num_abstract = len(lines)
    for i, line in enumerate(lines):
        abstract, v_category = line.strip().split(',')
        print(abstract)
        if v_category == "Human":
            num_human += 1
        elif v_category == "Paraphrased":
            num_paraphrased += 1
        elif v_category == "GPT":
            num_gpt += 1
        result, words, bigrams = analyze_abstract(abstract)
        num_mentions += result['MENTIONS']
        max_mentions = max(max_mentions, result['MENTIONS'])
        num_pos_emojis += result['POS_EMOS']
        num_neg_emojis += result['NEG_EMOS']
        max_emojis = max(max_emojis, result['POS_EMOS'] + result['NEG_EMOS'])
        num_urls += result['URLS']
        max_urls = max(max_urls, result['URLS'])
        num_words += result['WORDS']
        min_words = min(min_words, result['WORDS'])
        max_words = max(max_words, result['WORDS'])
        all_words.extend(words)
        num_bigrams += result['BIGRAMS']
        all_bigrams.extend(bigrams)
        write_status(i+1, num_abstract)
num_emojis = num_pos_emojis + num_neg_emojis
unique_words = list(set(all_words))
with open(urlfil+'data/processed_unique.txt', 'w') as uwf:
    uwf.write('\n'.join(unique_words))
num_unique_words = len(unique_words)
num_unique_bigrams = len(set(all_bigrams))
print ('\nCalculating frequency distribution')
# Unigrams
freq_dist = FreqDist(all_words)
pkl_file_name = urlfil+'data/processed_freqdist.pkl'
with open(pkl_file_name, 'wb') as pkl_file:
    pickle.dump(freq_dist, pkl_file)
print ('Saved uni-frequency distribution to %s' % pkl_file_name)
# Bigrams
bigram_freq_dist = get_bigram_freqdist(all_bigrams)
bi_pkl_file_name = urlfil+'/data/processed_freqdist-bi.pkl'
with open(bi_pkl_file_name, 'wb') as pkl_file:
    pickle.dump(bigram_freq_dist, pkl_file)
print ('Saved bi-frequency distribution to %s' % bi_pkl_file_name)
print ('\n[Analysis Statistics]')
print ('abstract => Total: %d, Human: %d, Paraphrased: %d, GPT: %d' % (num_abstract, num_human, num_paraphrased, num_gpt))
print ('User Mentions => Total: %d, Avg: %.4f, Max: %d' % (num_mentions, num_mentions / float(num_abstract), max_mentions))
print ('URLs => Total: %d, Avg: %.4f, Max: %d' % (num_urls, num_urls / float(num_abstract), max_urls))
print ('Emojis => Total: %d, Positive: %d, Negative: %d, Avg: %.4f, Max: %d' % (num_emojis, num_pos_emojis, num_neg_emojis, num_emojis / float(num_abstract), max_emojis))
print ('Words => Total: %d, Unique: %d, Avg: %.4f, Max: %d, Min: %d' % (num_words, num_unique_words, num_words / float(num_abstract), max_words, min_words))
print ('Bigrams => Total: %d, Unique: %d, Avg: %.4f' % (num_bigrams, num_unique_bigrams, num_bigrams / float(num_abstract)))


# In[ ]:




