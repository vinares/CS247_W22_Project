import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model, save_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D ,LSTM ,TimeDistributed
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping , ModelCheckpoint
from keras.layers import Dropout
import re
import os
import pandas as pd
test_dir = "D:/Work/UCLA/2022WINTER/CS247/Project/Data/test/"
TRAIN_CSV = "D:/Work/UCLA/2022WINTER/CS247/Project/Data/train.csv"
MODEL_SAVE = "model_pre.h5"

train = pd.read_csv(TRAIN_CSV)
STOPWORDS = ["a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are", "as", "at", "be",
             "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
             "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
             "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
             "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves"]
train = train.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ',
                                   text)  # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('',
                              text)  # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = text.replace('x', '')
    #    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwords from text
    return text


train['discourse_text'] = train['discourse_text'].apply(clean_text)
train['discourse_text'] = train['discourse_text'].str.replace('\d+', '')

lens_list = [len(i.split()) for i in train['discourse_text'] ]
max_length = max(lens_list)
max_ind =lens_list.index(max_length)
print('index of maximum lenght(longer sentence): ' ,max_ind )
print('maximum lenght is : ', max_length)

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 10000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = max_length
# This is fixed.
EMBEDDING_DIM = 300
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(train['discourse_text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(train['discourse_text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

y_label = pd.get_dummies(train['discourse_type'])
Y = y_label.values
print('Shape of label tensor:', Y.shape)

label_names = y_label.columns

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(10, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 2
batch_size = 64

model.fit(X, Y,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.1,
          callbacks=[ModelCheckpoint('best_model.h5', save_best_only = True)])

test_files = os.listdir(test_dir)

for file in range(len(test_files)):
    test_files[file] = str(test_dir) + "/" +  str(test_files[file])

print("Total number of test files = " , len(test_files))

test_names, test_texts = [], []
for f in list(os.listdir(test_dir)):
    test_names.append(f.replace('.txt', ''))
    test_texts.append(open(test_dir + f, 'r').read())

test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})

discourse_id = []
discourse_text = []
discourse_start =[]
discourse_end = []
predictionstring = []
for text in range(len(test_texts.text)):
    doc = test_texts.text[text]
    paras = re.split(r'[.]\n',doc)
    start = 0
    for para in paras:
        positions = re.findall(r'\w+[.]', para)
        if len(positions) <=2 :
            txts = re.split(r'[.]\s',para)
            for i in txts:
                discourse_text.append(i)
                length = len(i.split())
                end = start + length
                l = list(range(start+1,end+1 ))
                l = [str(j) for j in l]
                l = ' '.join(l)
                predictionstring.append(l)
                discourse_start.append(start+1)
                discourse_end.append(end)
                discourse_id.append(test_texts.id[text])
                start += length

        else:
            if len(positions)%2 == 0:
                split_pos = int(len(positions)/2)
                split_word = positions[split_pos]
            else :
                split_pos = int((len(positions)+1)/2)
                split_word = positions[split_pos]

            words = para.split(' ')
            position = words.index(split_word)
            part1 = words[:position]
            part2 = words[position:]
            part1 = ' '.join(part1)
            part2 = ' '.join(part2)
            parts = [part1 ,part2]
            for i in parts:
                discourse_text.append(i)
                length = len(i.split())
                end = start + length
                l = list(range(start+1,end+1 ))
                l = [str(k) for k in l]
                l = ' '.join(l)
                predictionstring.append(l)
                discourse_start.append(start+1)
                discourse_end.append(end)
                discourse_id.append(test_texts.id[text])
                start += length

testing_data =pd.DataFrame()
testing_data['discourse_id'] =discourse_id
testing_data['discourse_text'] = discourse_text
testing_data['discourse_start'] = discourse_start
testing_data['discourse_end'] = discourse_end
testing_data['predictionstring'] = predictionstring
testing_data.head()

testing_data['test_sentences'] = testing_data['discourse_text'].apply(clean_text)
testing_data['test_sentences'] = testing_data['discourse_text'].str.replace('\d+', '')

X_test = tokenizer.texts_to_sequences(testing_data['test_sentences'].values)
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_test.shape)

y_pred = model.predict(X_test)
output = pd.DataFrame(y_pred ,columns= label_names)
output = list(output.idxmax(axis = 1))
submission_df = pd.DataFrame()
submission_df['id'] = testing_data['discourse_id']
submission_df['class'] = output# label of y_predict
submission_df['predictionstring'] = testing_data['predictionstring']
mapping = { 1:'Claim' , 2:'Evidence' ,  3:'Position' , 4:'Concluding Statement' , 5:'Lead', 6:'Counterclaim', 7:'Rebuttal' }
submission_df['class']= submission_df['class'].replace(mapping)

submission_df['class'].unique()

submission_df.to_csv("submission.csv", index=False)

model.save(MODEL_SAVE)