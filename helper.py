import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(), 
                                                           s['POS'].values.tolist(), 
                                                           s['Tag'].values.tolist())]
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None
#getter = SentenceGetter(df)
#sentences = getter.sentences

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        #'word[-3:]': word[-3:],
        #'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    
    if i > 1:
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:postag': postag2,
            '-2:postag[:2]': postag2[:2],
        })

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    
    if i < len(sent)-2:
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        features.update({
            '+2:word.lower()': word2.lower(),
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:postag': postag2,
            '+2:postag[:2]': postag2[:2],
        })

    return features


def word2features_casual(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        #'word[-3:]': word[-3:],
        #'word[-2:]': word[-2:],
        #'word.isupper()': word.isupper(),
        #'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            #'-1:word.istitle()': word1.istitle(),
            #'-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    
    if i > 1:
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        features.update({
            '-2:word.lower()': word2.lower(),
            #'-2:word.istitle()': word2.istitle(),
            #'-2:word.isupper()': word2.isupper(),
            '-2:postag': postag2,
            '-2:postag[:2]': postag2[:2],
        })

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            #'+1:word.istitle()': word1.istitle(),
            #'+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    
    if i < len(sent)-2:
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        features.update({
            '+2:word.lower()': word2.lower(),
            #'+2:word.istitle()': word2.istitle(),
            #'+2:word.isupper()': word2.isupper(),
            '+2:postag': postag2,
            '+2:postag[:2]': postag2[:2],
        })

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2features_casual(sent):
    return [word2features_casual(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]
    
def turn2correctdf(input):
    real_df = pd.DataFrame(input)
    real_df.insert(0, "Sentence #", "Sentence: 0")
    real_df.insert(3, "Tag", "None")
    real_df.rename(columns={0: 'Word', 1 : 'POS'}, inplace=True)
    return real_df

def turndf2modelInput(df_in):
    new_getter = SentenceGetter(df_in)
    new_sentence = new_getter.sentences
    for_predict = [sent2features(s) for s in new_sentence]  
    return for_predict

def turndf2modelInput_casual(df_in):
    new_getter = SentenceGetter(df_in)
    new_sentence = new_getter.sentences
    for_predict = [sent2features_casual(s) for s in new_sentence]  
    return for_predict

def fromString2modelInput(str_in):
    return turndf2modelInput(turn2correctdf(nltk.pos_tag(word_tokenize(str_in))))

def fromString2modelInput_casual(str_in):
    return turndf2modelInput(turn2correctdf_casual(nltk.pos_tag(word_tokenize(str_in))))


