import numpy as np
import random
import os
import sys
import re
import matplotlib.pylab as plt
from random import seed, shuffle, randint
from string import digits
import csv



HTTP_RE = re.compile(r"ST@RT.+?INFO\s+(.+?)\s+END", re.MULTILINE | re.DOTALL)

'''
________________________________________________________________________________
Utilities for the Seq2Seq-Autoencoder
'''


def http_re(data):
    """
    Extracts HTTP requests from raw data string in special logging format.
    Logging format `ST@RT\n%(asctime)s %(levelname)-8s\n%(message)s\nEND`
    where `message` is a required HTTP request bytes.
    """
    return HTTP_RE.findall(data)


def get_requests_from_file(path):
    """
    Reads raw HTTP requests from given file.
    """
    with open(path, 'r') as f:
        file_data = f.read()
    requests = http_re(file_data)
    return requests


def get_hack_data():
    words =[]
    data = get_requests_from_file("C:/Users/Andreas/Desktop/seq2seq - continous/data/vulnbank_train.txt")
    for k in data:
        for i in k:
            words.append(i)

    words.append("<GO>")
    words.append("<NAN>")
    words.append("<EOS>")
    words.append("<PAD>")
    words = set(words)
    word2int = {}
    int2word = {}
    vocab_size = len(words)
    print("vocab_size")
    print(vocab_size)
    for i,word in enumerate(words,0):
        word2int[word] = i
        int2word[i] = word
    data = generate_sentence_int(data,word2int)
    return  word2int, int2word,vocab_size,data



def shuffle_data(training_data):
     np.random.shuffle(training_data)
     return training_data

def shuffle_batch(text,label):
    seed(randint(1,1000))
    shuffle(text)
    shuffle(label)
    return text, label

def split_dat(text,label,ratio):
    text_len = len(text)
    text,label = shuffle_batch(text,label)
    test_size = text_len * ratio
    test_size = int(test_size)
    print("test_size")
    print(test_size)
    training_text = text[0:test_size]
    training_label = label[0:test_size]
    test_text = text[test_size:]
    test_label = label[test_size:]
    return training_text,training_label,test_text,test_label

def get_labels():
    labels = []
    for file in os.listdir("C:/Users/Andreas/Desktop/Text_clas/DOM_Email_Classification_KI_Pilot"):
        if file == "label.txt":
            f = open(os.path.join("DOM_Email_Classification_KI_Pilot","label_2.txt"),"r",errors = "ignore")
            for line in f:
                line = line.replace("\n","")
                labels.append(line)
    labels_len = set(labels)
    return labels_len, labels

def labels_one_hot(labels_len,labels):
    zero_vec = np.zeros(len(labels_len))
    label2vec = []
    one_hot_vector = []
    for i in range(0,len(labels_len)):
        test = list(zero_vec)
        test[i]= 1
        one_hot_vector.append(test)
    labels_len = list(labels_len)
    for i in range(0,len(labels)):
        k = labels_len.index(labels[i])
        vec = one_hot_vector[k]
        label2vec.append(vec)
    return label2vec

def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    # get '<GO>' id source: https://gist.github.com/deep-diver/460e12e5ff14585cc86ebf3118d30075#file-process_decoder_input-py
    go_id = target_vocab_to_int['<GO>']
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat( [tf.fill([batch_size, 1], go_id), after_slice], 1)

    return after_concat

def clear_text(text):
    text = re.sub(r"[A-Za-z0-9(),!?\'\'\"]"," ",text)
    text = re.sub(r"\s{2,}"," ", text)
    text = text.lower()
    return text

def read_training_data():
    words =[]
    epoch = []
    counter = 1
    training_data=[]
    for file in os.listdir("C:/Users/Andreas/Desktop/seq2seq - continous/data"):
        if file == "honeypot.txt":
            with open(os.path.join("data","honeypot.txt"),"r",encoding='utf-8',errors = "ignore") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\t')
                for line in csv_reader:
                    training_data.append(line)
                    #line = line.replace("\n","")
                    #line = line.replace("\t","")
                    #line = line.lower()
                #line = clear_text(line)
                '''


                for x in line.split():
                    remove_digits = str.maketrans('', '', digits)
                    x = x.translate(remove_digits)
                    x = re.sub('[^ a-zA-Z0-9]', '', x)
                    if x != "":
                        words.append(x)
                if line != "":
                    training_data.append(line)
    words.append("<GO>")
    words.append("<NAN>")
    words.append("<EOS>")
    words.append("<PAD>")
    words = set(words)
    word2int = {}
    int2word = {}
    vocab_size = len(words)
    print("vocab_size")
    print(vocab_size)
    for i,word in enumerate(words,0):
        word2int[word] = i
        int2word[i] = word
    return  word2int, int2word,vocab_size,training_data
'''
    return training_data
def getsentencce(sentence,int2wor):
    new_sentence=[]
    for i in range(0,len(sentence)):
        x = int2wor[sentence[i]]
        new_sentence.append(x)

    return new_sentence

def load_test_data(word2int):
    words =[]
    epoch = []
    counter = 1
    training_data=[]
    for file in os.listdir("C:/Users/Andreas/Desktop/seq2seq/data"):
        if file == "anomaly.txt":
            f = open(os.path.join("data","anomaly.txt"),"r",encoding='utf-8',errors = "ignore")
            for line in f:
                line = line.replace("\n","")
                line = line.replace("\t","")
                line = line.lower()
                #line = clear_text(line)
                for x in line.split():
                    remove_digits = str.maketrans('', '', digits)
                    x = x.translate(remove_digits)
                    x = re.sub('[^ a-zA-Z0-9]', '', x)
                    words.append(x)
                if line != "":
                    training_data.append(line)
    training_data = generate_sentence_int(training_data,word2int)
    return training_data


def pad_sequences(text,word2int):
    max_len= len(max(text,key=len))
    l=[]
    for i in range(0,len(text)):
        l.append(max_len)
        for j in range(len(text[i]),max_len):
            text[i].append(word2int["<PAD>"])
    l = np.asarray(l,dtype="int32")
    return text,l

def _process_request(req):
    vocab=Vocabulary()
    seq = vocab.string_to_int(req)
    l = len(seq)
    return seq, l

def generate_training_data_sentence():
    for file in os.listdir("C:/Users/Andreas/Desktop/seq2seq/data"):
        if file == "book.txt":
            with  open(os.path.join("data","book.txt"),"r",errors = "ignore") as f:
                text = f.read()

            sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)

    return sentences


def generate_training_data():
    for file in os.listdir("C:/Users/Andreas/Desktop/seq2seq/data"):
        if file == "book.txt":
            f = open(os.path.join("data","book.txt"),"r",errors = "ignore")
            first_colum = [row[0] for row in csv.reader(f,delimiter ='\t')]
            for i in range(0,len(first_colum)):
                first_colum[i] = first_colum[i].split(" ")

                for f in range(0,len(first_colum[i])):
                    first_colum[i][f] = first_colum[i][f].replace("\n","")
                    first_colum[i][f] = first_colum[i][f].replace("\t","")
    print(first_colum[1])
    return first_colum

def generate_sentence_int(raw_sentences,word2int):
    sentenc_int=[]
    for i in range(0,len(raw_sentences)):
        raw_sentences[i] = raw_sentences[i].split()
        singe_sentenc = []
        for j in raw_sentences[i]:
            for l in j:
                if l in word2int.keys():
                    singe_sentenc.append(word2int[l])
                else:
                    singe_sentenc.append(word2int["<NAN>"])
        singe_sentenc.append(word2int["<EOS>"])
        sentenc_int.append(singe_sentenc)
    return sentenc_int

def generate_data_with_label():
    raw_sentences = generate_training_data()
    word2int, int2word, vocab_size = read_training_data()
    test = generate_sentence_int(raw_sentences,word2int)
    max = 0
    index = 0
    for i in range(0,len(test)):
        if len(test[i])>= max:
            max = len(test[i])
            index = i
    test = pad_sequences(test,max)
    label_len,labels = get_labels()
    one_hot = labels_one_hot(label_len,labels)
    text_training, label_training, text_test, label_test = split_dat(test,one_hot,0.95)
    text_training = np.asarray(text_training,dtype=np.int32)
    label_training = np.asarray(label_training,dtype=np.float32)
    text_test = np.asarray(text_test,dtype=np.int32)
    label_test = np.asarray(label_test,dtype=np.float32)

    return text_training, label_training, text_test, label_test, vocab_size, max

def generate_data_no_label():
    raw_sentences = generate_training_data()
    word2int, int2word, vocab_size = read_training_data()
    training_data = generate_sentence_int(raw_sentences,word2int)
    max = 0
    index = 0
    for i in range(0,len(training_data)):
        if len(training_data[i])>= max:
            max = len(training_data[i])
            index = i


    return training_data, word2int, int2word

def generate_sentece():
    word2int, int2word,vocab_size,training_data =read_training_data()
    training_data = generate_sentence_int(training_data,word2int)
    return word2int, int2word,vocab_size,training_data


word2int, int2word,vocab_size,training_data =  get_hack_data()


'''
data = get_requests_from_file("C:/Users/Andreas/Desktop/seq2seq - continous/data/anomaly.txt")
print(data)
data = generate_sentence_int(data,word2int)
batched_test_data,l = pad_sequences(data,word2int)
print(batched_test_data)
batched_test_data = np.asarray(batched_test_data,dtype="int32")
example = getsentencce(batched_test_data[0],int2word)
print(example)
ba_si=1
size = l[0]
'''
