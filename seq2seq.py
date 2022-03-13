# -*- coding: utf-8 -*-
"""
    Haiying's lstm encoder-decoder
    The customized constrained-beam-search decoder is not finished
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from google.colab import drive
#TRAIN_CSV = "data/train.csv"
#TRAIN_DIR = "data/train"
drive.mount('/content/drive')
TRAIN_CSV = "/content/drive/MyDrive/Colab Notebooks/ori_data/train.csv"
TRAIN_DIR = "/content/drive/MyDrive/Colab Notebooks/ori_data/train"

# Step 1: preprocess data
train = pd.read_csv(TRAIN_CSV)
IDS = train.id.unique()
print(f"Number of train samples are: {len(IDS)}")
sent0 = train.iloc[0]["discourse_text"]
print("sent0: ", sent0)

# 14 classes for NER
lead_b = 1
lead_i = 2
position_b = 3
position_i = 4
evidence_b = 5
evidence_i = 6
claim_b = 7
claim_i = 8
counterclaim_b = 9
counterclaim_i = 10
rebuttal_b = 11
rebuttal_i = 12
conclusion_b = 13
conclusion_i = 14
other = 15

targets_b = {'Lead':lead_b, 'Position':position_b, 'Evidence':evidence_b, 'Claim':claim_b, 
             'Counterclaim':counterclaim_b, 'Rebuttal':rebuttal_b, 'Concluding Statement':conclusion_b}
targets_i = {'Lead':lead_i, 'Position':position_i, 'Evidence':evidence_i, 'Claim':claim_i, 
             'Concluding Statement':conclusion_i, 'Counterclaim':counterclaim_i, 'Rebuttal':rebuttal_i}


# clean text
import re

def clean_text(text):
    #text = text.lower()  lowercase text
    #text = re.sub(r"([?.!,;])", r"\1 ", text) # pad punctuations
    #text = re.sub(r"[^\w\s]", '', text) # remove bad characters
    text = re.sub(r'\s+', ' ', text) # merge multiple spaces
    return text
     
sent0 = clean_text(sent0)
print("cleaned sent0: ", sent0)

print("Start processing documents...")
train_texts = {}
train_seqs = {}
train_labels = {}
train_lengths = {}
for id in IDS:
    filepath = Path(TRAIN_DIR) / f"{id}.txt"
    text = open(filepath, 'r').read().strip()
    words = text.split()
    train_lengths[id] = len(words)
    text = clean_text(text)
    train_texts[id] = text

# tokenize (build vocabulary)
MAX_SEQ_LEN = 512
texts = train_texts.values()
tokenizer = Tokenizer(filters='', oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index)
print(f"find tokens: {VOCAB_SIZE}")

# build input sequences
for id in IDS:
    text = train_texts[id]
    origin_len = train_lengths[id]
    seq = tokenizer.texts_to_sequences([text])[0]
    if len(seq) != origin_len:
        # make sure that sequence length does not change after tokenization
        raise RuntimeError(f" file: {id} seq len： {len(seq)} text len: {origin_len}")
    train_seqs[id] = seq

# build labels
train_labels = {}
for id in IDS:
    seq_len = train_lengths[id]
    labels = [other]*seq_len
    rows = train.loc[train['id'] == id]
    for i,row in rows.iterrows():
        label = row['discourse_type']
        label_b = targets_b[label]
        label_i = targets_i[label]
        prediction_string = row['predictionstring']
        indices = prediction_string.strip().split()
        indices = [int(idx) for idx in indices]
        labels[indices[0]] = label_b
        for idx in indices[1:]:
            if idx < seq_len:
                labels[idx] = label_i
            else:
                print(f"Error set label: file {id}, length {seq_len}, word index {idx}, label {label}")
    train_labels[id] = labels

print("Finish processing documents.")
id0 = IDS[0]
length0 = train_lengths[id0]
X0 = train_seqs[id0]
Y0 = train_labels[id0]
print("id0: ", id0)
print("length0: ", length0)
print("X0: ", X0)
print("y0: ", Y0)
#exit(1)

# build tf dataloader
MAX_SEQ_LEN = 512
BATCH_SIZE = 32
input_sequences = [train_seqs[id] for id in IDS]
output_sequences = [train_labels[id] for id in IDS]
input_sequences = pad_sequences(input_sequences, maxlen=MAX_SEQ_LEN, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=MAX_SEQ_LEN, padding='post')
print("inputs: ", input_sequences.shape)
#print("input0: ", input_sequences[0])
print("outputs: ", output_sequences.shape)
#print("output0: ", output_sequences[0])

train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, output_sequences)).batch(BATCH_SIZE, drop_remainder=True)
X0, Y0 = next(train_dataset.as_numpy_iterator())
print("input0: ",  X0.shape)
print("output0:", Y0.shape)

# Encoder model
EMBED_SIZE = 300
HIDDEN_SIZE = 1024
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)

    ##-------- LSTM layer in Encoder ------- ##
    self.lstm_layer = tf.keras.layers.LSTM(self.hidden_size,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x):
    x = self.embedding(x)
    output,h, c = self.lstm_layer(x)
    return output, h, c

# test encoder
encoder = Encoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
encoder_output, h, c = encoder(X0)
print("Output: {}".format(encoder_output.shape))
print("hidden: {}".format(h.shape))
print("cell: {}".format(c.shape))

!pip install tensorflow_addons

import tensorflow_addons as tf_addons

# Implement basic LSTM decoder
OUTPUT_SIZE = 15 # fourteen NER CLASSES 
class Decoder(tf.keras.Model):
  def __init__(self, hidden_size, output_size, batch_size):
    super(Decoder, self).__init__()
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.linear = tf.keras.layers.Dense(self.output_size) # map lstm outputs to class probabilitlies
    self.lstm_cell = tf.keras.layers.LSTMCell(self.hidden_size)
    self.sampler = tf_addons.seq2seq.sampler.TrainingSampler() # sampler for decoder
    self.decoder = tf_addons.seq2seq.BasicDecoder(self.lstm_cell, sampler=self.sampler, output_layer=self.linear)


  def build_initial_state(self, batch_size, encoder_state, Dtype):
    decoder_initial_state = self.lstm_cell.get_initial_state(batch_size=batch_size, dtype=Dtype)
    print("initial state: ", decoder_initial_state)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
    return decoder_initial_state


  def call(self, inputs, initial_state):
    x = tf.one_hot(inputs, depth=self.output_size, dtype=tf.float32) # map labels to one-hot 
    outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_size*[MAX_SEQ_LEN])
    return outputs

# test decoder
decoder = Decoder(HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE)
#decoder = Decoder1(OUTPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE)
#initial_state = decoder.build_initial_state(BATCH_SIZE, [h, c], tf.float32)
START = tf.zeros(shape=(BATCH_SIZE, 1), dtype=tf.int32)
inputs = tf.concat((START, Y0[:,1:]), axis=1)
decoder_outputs = decoder(inputs, initial_state=[h,c])
print("decoder output: ", decoder_outputs.rnn_output.shape)

# TODO: define Soft-Constrained-Beam-Search-Decoder that integrates soft constraints 
#       on output structures into the beam search process

class SoftConstrainBeamSearchDecoder():
    pass