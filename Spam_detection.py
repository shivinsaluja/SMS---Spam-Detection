import pandas as pd
import math
import spacy
import scipy
from os import sys
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Model,model_from_json
from keras.layers import Dense, Dropout, LSTM, Input,add
from keras.utils import np_utils
import keras.backend as K
from nltk.corpus import stopwords
from nltk.tokenize import  word_tokenize
from pathlib import Path
import pandas as pd
import keras.callbacks as kc
import scipy
import csv
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import scipy.sparse
import random
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import re
import en_core_web_sm

nlp = en_core_web_sm.load()
nlp_synonyms = en_core_web_sm.load()
pos_set={'NN','NNS','NNP','NNPS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ','RB','RBR','RBS'}


class ProcessDataset(object):
    
    def __init__(self):


        All_statements = []
        All_statements_vectors = []
        X_all_doc_vec = []
        
        All_Intents = []
        
        file = pd.read_csv('spam_final.csv',encoding='latin-1')
        
        sentence = file['Text']
        labellist = file['Label'];

        for x_sent,label in zip(sentence,labellist):
                x_sent = str(x_sent)
                if len(x_sent) > 0:
                    x_sent = x_sent.lower()
                    word_tokens = word_tokenize(x_sent)
                    
                    tokenized_sentence = [w for w in word_tokens]
                    #tokenized_sentence.sort()
                    x_sent = ""
                    
                    for i in tokenized_sentence:
                        
                        x_sent = x_sent+str(i).strip()+" "

                    x_doc = nlp(str(x_sent))
                    
                    x_doc_vec = x_doc.vector/x_doc.vector_norm
                    x_vec_seq = []
                    
                    for word in x_doc:
                        x_vec_seq.append(word.vector/word.vector_norm)
                    
                    x_vec_seq = np.array(x_vec_seq)
                    All_statements.append(x_sent)
                    X_all_doc_vec.append(x_doc_vec)
                    All_statements_vectors.append(x_vec_seq)
                    All_Intents.append(label)



        self.All_statements = All_statements
        self.All_statements_vectors = All_statements_vectors
        self.X_all_doc_vec = X_all_doc_vec
        self.All_Intents = All_Intents


def filter_word(word,postag):
    if postag in pos_set:
        return word
    else:
        return ''

def pad_vec_sequences(sequences,maxlen=20):
    new_sequences = []
    for sequence in sequences:
        orig_len, vec_len = np.shape(sequence)
        if orig_len < maxlen:
            new = np.zeros((maxlen,vec_len))
            new[maxlen-orig_len:,:] = sequence
        else:
            new = sequence[orig_len-maxlen:,:]
            
        new_sequences.append(new)
    return np.array(new_sequences)
    
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def remove_special_characters(question):
    question = re.sub(r'[^\w]', ' ', question)
    question = re.sub(' +',' ',question)
    question = re.sub(r"n\'t", " not", question)
    question = re.sub(r"\'re", " are", question)
    question = re.sub(r"\'s", " is", question)
    question = re.sub(r"\'d", " would", question)
    question = re.sub(r"\'ll", " will", question)
    question = re.sub(r"\'t", " not", question)
    question = re.sub(r"couldnt ", " could not ", question)
    question = re.sub(r"thats ", " that is ", question)
    question = re.sub(r"cant ", " can not ", question)
    question = re.sub(r"dont ", " do not ", question)
    question = re.sub(r" im ", " i am ", question)
    question = re.sub(r"5lb ", " ", question)
    question = re.sub(r"15g ", " ", question)
    question = re.sub(r"shouldve ", " should have ", question)
    question = re.sub(r"wont ", " will not ", question)
    question = re.sub(r" hes ", " he is ", question)
    question = re.sub(r"ive ", " i have ", question)
    question = re.sub(r"wouldnt ", " would not ", question)
    question = re.sub(r"8pm ", " ", question)
    question = re.sub(r"8am", " ", question)
    question = re.sub(r"9pm", " ", question)
    question = re.sub(r"10am", " ", question)
    question = re.sub(r"9am", " ", question)
    question = re.sub(r"1pm", " ", question)
    question = re.sub(r"11pm", " ", question)
    question = re.sub(r" s ", " is ", question)
    question = re.sub(r"wasn t ", " was not ", question)
    question = re.sub(r"don t ", " do not ", question)
    question = re.sub(r"didn t ", " did not ", question)
    question = re.sub(r"doesnt", " does not ", question)
    question = re.sub(r"havent", " have not ", question)
    question = re.sub(r"wan", " want ", question)
    question = re.sub(r"whats", " what is ", question)
    question = re.sub(r" id ", " identification ", question)
    question = re.sub(r" me ", " ", question)
    question = re.sub(r" ì_ " ," ", question)
    return question


nb_classes = 2
epochs = 30


def clean_dataset():
     file=pd.read_csv('spam.csv',encoding='latin-1')
     f = open('spam_final.csv', 'w', newline='')
     writer = csv.writer(f)
     
     sentence = file['v2']
     labellist = file['v1']
     writer.writerow(['Text','Label'])
     j=0
     for x_sent,label in zip(sentence,labellist):
            if len(x_sent) > 0:
                x_sent = x_sent.lower()
                x_sent = remove_special_characters(x_sent)
                j = j+1
                print(j)
                print(x_sent)
                
                word_tokens = nltk.word_tokenize(x_sent)
                filtered_sentence = [w for w in word_tokens]
                
                new_filtered_sentence=[]
                
                token_map = {}
                token_map = nltk.pos_tag(word_tokens)
                token_dictionary={}
                
                for i in token_map:
                    token_dictionary[str(i[0])] = str(i[1])

                for i in filtered_sentence:
                   temp = filter_word(str(i),str(token_dictionary[str(i)]))
                   new_filtered_sentence.append(temp)
                    
                new_sent=""
                for i in new_filtered_sentence:
                    new_sent = new_sent + str(i).strip() +" "
                    
                new_sent = remove_special_characters(new_sent)
                new_sent = new_sent.strip()
                if (label=='ham'):
                    writer.writerow([new_sent,1])
                else:
                    writer.writerow([new_sent,0])
                    
                    
clean_dataset()

check_dataset = []
check_intents =[]
ds = ProcessDataset()
X_all = (ds.All_statements_vectors)
All_Intents = ds.All_Intents

temp_1 = ds.All_statements



def train_dataset():
    
    batch_size = 3
    processed_dataset = ProcessDataset()
    print("Dataset Processed")
    X_all = pad_vec_sequences(processed_dataset.All_statements_vectors)
    check_dataset = X_all
    print(check_dataset)
    All_Intents = processed_dataset.All_Intents
    check_Intents = All_Intents
    
    Labels = np_utils.to_categorical(All_Intents)
    x_train, x_test, y_train, y_test = train_test_split(X_all, Labels, test_size=0.2)
    
    print("Training for the very first time")
    max_len = 20
    hidden_dim = 300
    K.clear_session()
    
    sequence = Input(shape=(max_len,384), dtype='float32')
    
    forwards = LSTM(hidden_dim,dropout=0.1, recurrent_dropout=0.1)(sequence)
    
    backwards = LSTM(hidden_dim,dropout=0.1, recurrent_dropout=0.1,go_backwards=True)(sequence)
    merged = add([forwards, backwards])
    after_dp = Dropout(0.1)(merged)
    output = Dense(nb_classes, activation='softmax')(after_dp)
    model=Model(inputs=sequence, outputs=output)
    model.compile('adam', 'categorical_crossentropy',metrics=['accuracy', mean_pred])

    ES = EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0, mode='auto')
    model.fit(x_train, y_train,batch_size=batch_size,nb_epoch=epochs,validation_data=[x_test, y_test],callbacks=[ES])
    model_json = model.to_json()
    
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        model.save_weights("model.h5")
        print("Saved model to disk")
        
train_dataset()


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    global model
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    global graph
    graph = tf.get_default_graph()


def TestModel(x_sent):
    
    load_model()
    x_sent = x_sent.lower()
    x_sent = remove_special_characters(x_sent)
    
    word_tokens = word_tokenize(x_sent)
    tokenized_sentence = [w for w in word_tokens]

    new_filtered_sentence=[]
    word_tokens_spacy = nlp(x_sent)
    token_map = {}
                
    for tok in word_tokens_spacy:
        token_map[str(tok.text)] = str(tok.tag_)
        
    for i in tokenized_sentence:
        temp = filter_word(i,token_map[str(i)])
        new_filtered_sentence.append(temp)
                    
    new_sent=""
    for i in new_filtered_sentence:
        new_sent = new_sent + str(i).strip() +" "
                    
    new_sent = remove_special_characters(new_sent)
    new_sent = new_sent.strip()
    

    x_sent = new_sent
    print(x_sent)
    
    All_statements = []
    check = []
    X_all_doc_vec = []
    x_doc = nlp(str(x_sent))
    x_doc_vec = x_doc.vector/x_doc.vector_norm
    x_vec_seq = []
    
    for word in x_doc:
        
        x_vec_seq.append(word.vector/word.vector_norm)
        
        
    x_vec_seq = np.array(x_vec_seq)
    All_statements.append(x_sent)
    X_all_doc_vec.append(x_doc_vec)
    check.append(x_vec_seq)
    check = pad_vec_sequences(check)
    
    
    with graph.as_default():
        
        ynew = np.argmax((model.predict(check)))
        check_prob = model.predict(check)
        print(check_prob)
        

        if ynew==0:
            print("This is a spam sms")
        else:
            print("This is not a spam sms")
            
            
question="Arriving Today: Official GRE Verbal Reasoning Practice Questions is out for delivery"
TestModel(question)


     
