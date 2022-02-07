import os
import math

# These first two functions require os operations and so are completed for you
# Completed for you


def load_training_data(vocab, directory):
    ''' Create the list of dictionaries '''
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

# Completed for you


def create_vocabulary(directory, cutoff):
    ''' Create a vocabulary from the training directory
        return a sorted vocabulary list
    '''

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f, 'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

# The rest of the functions need modifications ------------------------------
# Needs modifications


def create_bow(vocab, filepath):
    ''' Create a single dictionary for the data
        Note: label may be None
    '''
    bow = {}
    # TODO: add your code here
    store = {}
    noneCount = 0

    for v in vocab:
        store[v] = 0
    file = open(filepath, 'r')
    for line in file:
        for word in line.split():
            ifFound = False
            for vword in vocab:
                if vword == word:
                    store[word] += 1
                    ifFound = True
            if ifFound == False:
                noneCount += 1

    for element in store:
        if store[element] != 0:
            bow[element] = store[element]
    if noneCount != 0:
        bow[None] = noneCount
    return bow

# Needs modifications


def prior(training_data, label_list):
    ''' return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    '''

    smooth = 1  # smoothing factor
    logprob = {}
    for label in label_list:
        count = 0
        for data in training_data:
            if data['label'] == label:
                count += 1
        logprob[label] = math.log((count + 1) / (len(training_data) + 2))
    # TODO: add your code here

    return logprob

# Needs modifications


def p_word_given_label(vocab, training_data, label):
    ''' return the class conditional probability of label over all words, with smoothing '''

    smooth = 1  # smoothing factor
    word_prob = {}
    word_count = {}
    totalCount = 0
    for word in vocab:
        word_count[word] = 0
        word_count[None] = 0
    for data in training_data:
        if data['label'] == label:
            for word in data['bow']:
                word_count[word] += data['bow'][word]
                totalCount += data['bow'][word]
    for word in vocab:
        word_prob[word] = math.log(
            (word_count[word] + smooth*1) / (totalCount + smooth*(len(vocab) + 1)))

    word_prob[None] = math.log(
        (word_count[None] + smooth*1) / (totalCount + smooth*(len(vocab) + 1)))
    # TODO: add your code here

    return word_prob


##################################################################################
# Needs modifications
def train(training_directory, cutoff):
    ''' return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    '''
    retval = {}
    label_list = os.listdir(training_directory)
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    retval['vocabulary'] = vocab
    retval['log prior'] = prior(training_data, ['2020', '2016'])
    retval['log p(w|y=2016)'] = p_word_given_label(
        vocab, training_data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(
        vocab, training_data, '2020')
    # TODO: add your code here

    return retval

# Needs modifications


def classify(model, filepath):
    ''' return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    '''
    retval = {}
    file = open(filepath, 'r')
    count2016 = model['log prior']['2016']
    count2020 = model['log prior']['2020']

    tempInt = 0
    for line in file:
        for word in line.split():
            ifFound = False
            for voc in model['vocabulary']:
                if voc == word:
                    ifFound = True
                    tempInt = model['log p(w|y=2016)'][word]
                    count2016 += tempInt
                    tempInt = model['log p(w|y=2020)'][word]
                    count2020 += tempInt
            if ifFound == False:
                tempInt = model['log p(w|y=2016)'][None]
                count2016 += tempInt
                tempInt = model['log p(w|y=2020)'][None]
                count2020 += tempInt

    retval['log p(y=2020|x)'] = count2020
    retval['log p(y=2016|x)'] = count2016

    if count2016 > count2020:
        retval['predicted y'] = '2016'
    if count2016 < count2020:
        retval['predicted y'] = '2020'
    if count2016 == count2020:
        retval['predicted y'] = 'DRAW'

    # TODO: add your code here

    return retval
