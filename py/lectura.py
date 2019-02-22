import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import TweetTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import string
import re
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from collections import Counter 
import emoji
from gsitk.preprocess import pprocess_twitter
from nltk.corpus import subjectivity
from textblob import TextBlob
import codecs
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import sentiwordnet as swn

def load_data(path):    
    df = pd.read_csv(path, sep='\t')    
    return df

def lexicon_generation(df):
    hs = df['text'][df['HS'] == 1]
    hs_tokens = []
    hs_lexicon = []
    no_hs = df['text'][df['HS'] == 0]
    no_hs_tokens = []
    no_hs_lexicon = []

    for tweet in hs:
        for tk in simon_tokenizer(tweet):
            hs_tokens.append(tk)

    contador = Counter(hs_tokens)
    comunes = contador.most_common()[0:450]

    for tupla in comunes:
        hs_lexicon.append(tupla[0])

        
    for tweet in no_hs:
        for tk in simon_tokenizer(tweet):
        	no_hs_tokens.append(tk)
        
    contador = Counter(no_hs_tokens)
    comunes = contador.most_common()[0:450]

    for tupla in comunes:
        no_hs_lexicon.append(tupla[0])

    no_hs_lexicon = list(set(no_hs_lexicon) - set(hs_lexicon))
    lexicon = [hs_lexicon, no_hs_lexicon]

    return lexicon

def simon_tokenizer(doc):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    urls = re.compile(r'.http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    char = re.compile(r'^[a-zA-Z]$')
    #har= re.compile(r'[a-zA-Z]')
    punct = re.compile(r'[.,-,:,<,;,(,=,¿,!,¡]')
    ht = re.compile(r'http.')
    bar = re.compile(r'//*')
    pr = ["rt","@","http","https","'s",'...', 'english', 'translation','):',
          '. .', '..','2-5','<3',']:','“','”','’','. ...','___','__','=(','‘','—','°','¢','•','®','—','…',
          '... .','--->','–','»','«','£','-->','×','->','©','\n','™','¤', '˜', 'â', '\U0001f92c', '‡', '', '\x9d', 
          '\u200d', '\x81', '\x8f', '¸', '❤', '‰', '🤔', '\u2066', '\u2069', '🤷', '‚', '¬', '🤣', '‹', 'ª', '☺', '„',
         '´', '·', '🤘', '🤗', '✋', '‼', 'ºðÿ', '✊', '☑', '¶', '¥', '\x8d', '¯', '²ðÿ', '\xad', '❌', '↺', '¨', 'âƒ', 
         'ðÿž', 'ã', '・', '🤤', '🤧', 'œðÿ', 'º', '♡', '✌', '♥', '⬇', '✅', 'œ', '€', '🤢', '†', '⚫', 'ˆðÿ', 'ˆ' ,'µ', 
         '\u200b', '¾', '✝', '¼', 'ºhttp', '§', '🤓', '✖', '⚠', 'âœ', 'ðÿœ', 'ðÿœ²ðÿœ³ðÿœ', '❛', '❜', '☠', '🤸', '🤡', 
         '〝', '─', '〞', '🆗', '❣', '♠', '🤞', '🤑', '\U0001f92b', 'ツ', '⚡', '🥂', '🤕', '❀', 'ᴴᴰ', '¦', '➡', '¿', '♀',
          '🤦', 'xd', '\U0001f92e', '\u2060', '♂', '⚽', '✨', '▶', '\U0001f92a', '⚔', '―', '♾', '►', '🧀', '\U0001f970', 
         '\U0001f9d0', '☝', '❓','🆚', '\U0001f92f', '͡', '🥚', 'xk', '🤙', 'ʖ', '͜', '⤵']
    
    
    tknzr = TweetTokenizer(reduce_len=True, strip_handles=True)
    tokens = tknzr.tokenize(doc.lower())   
    punctuation = set(string.punctuation)
    tokens_punct = [w for w in tokens if  w not in punctuation]
    tokens_clean = [w.replace('#', '') for w in tokens_punct if not urls.search(w) if w not in pr 
            if not bar.search(w) if not ht.search(w) if not char.search(w) if not punct.search(w)
            if not w.isdigit() if not emoji_pattern.search(w)]
    
    return tokens_clean

class TextTransformer(BaseEstimator, TransformerMixin):

    def tokenize_doc(self, doc):
        return simon_tokenizer(doc)
    
    def fit(self, docs, y=None):
        return self

    def transform(self, docs):
        return [self.tokenize_doc(doc) for doc in docs]

def english_tokenizer(words):
    """Preprocessing tokens as seen in the lexical notebook"""
    # Clean urls, punct, strange characters
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    urls = re.compile(r'.http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    char = re.compile(r'^[a-zA-Z]$')
    #har= re.compile(r'[a-zA-Z]')
    punct = re.compile(r'[.,-,:,<,;,(,=,¿,!,¡]')
    ht = re.compile(r'http.')
    bar = re.compile(r'//*')
    pr = ["rt","@","http","https","'s",'...', 'english', 'translation','):',
          '. .', '..','2-5','<3',']:','“','”','’','. ...','___','__','=(','‘','—','°','¢','•','®','—','…',
          '... .','--->','–','»','«','£','-->','×','->','©','\n','™','¤', '˜', 'â', '\U0001f92c', '‡', '', '\x9d', 
          '\u200d', '\x81', '\x8f', '¸', '❤', '‰', '🤔', '\u2066', '\u2069', '🤷', '‚', '¬', '🤣', '‹', 'ª', '☺', '„',
         '´', '·', '🤘', '🤗', '✋', '‼', 'ºðÿ', '✊', '☑', '¶', '¥', '\x8d', '¯', '²ðÿ', '\xad', '❌', '↺', '¨', 'âƒ', 
         'ðÿž', 'ã', '・', '🤤', '🤧', 'œðÿ', 'º', '♡', '✌', '♥', '⬇', '✅', 'œ', '€', '🤢', '†', '⚫', 'ˆðÿ', 'ˆ' ,'µ', 
         '\u200b', '¾', '✝', '¼', 'ºhttp', '§', '🤓', '✖', '⚠', 'âœ', 'ðÿœ', 'ðÿœ²ðÿœ³ðÿœ', '❛', '❜', '☠', '🤸', '🤡', 
         '〝', '─', '〞', '🆗', '❣', '♠', '🤞', '🤑', '\U0001f92b', 'ツ', '⚡', '🥂', '🤕', '❀', 'ᴴᴰ', '¦', '➡', '¿', '♀',
          '🤦', 'xd', '\U0001f92e', '\u2060', '♂', '⚽', '✨', '▶', '\U0001f92a', '⚔', '―', '♾', '►', '🧀', '\U0001f970', 
         '\U0001f9d0', '☝', '❓','🆚', '\U0001f92f', '͡', '🥚', 'xk', '🤙', 'ʖ', '͜', '⤵']
    
    
    tknzr = TweetTokenizer(reduce_len=True, strip_handles=True)
    tokens = tknzr.tokenize(words.lower())
    porter = PorterStemmer()
    lemmas = [porter.stem(t) for t in tokens]
    stoplist = stopwords.words('english')
    lemmas_clean = [w for w in lemmas if w not in stoplist]
    punctuation = set(string.punctuation)
    lemmas_punct = [w for w in lemmas_clean if  w not in punctuation]
    lemmas_clean = [w for w in lemmas_punct if not w.startswith('@') if w not in pr 
                if not bar.search(w) if not ht.search(w) if not char.search(w) if not punct.search(w)
                if not w.isdigit() if not emoji_pattern.search(w)]
    return lemmas_clean

def spanish_tokenizer(words):
    """Preprocessing tokens as seen in the lexical notebook"""
    # Clean urls, punct, strange characters
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    urls = re.compile(r'.http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    char = re.compile(r'^[a-zA-Z]$')
    #har= re.compile(r'[a-zA-Z]')
    punct = re.compile(r'[.,-,:,<,;,(,=,¿,!,¡]')
    ht = re.compile(r'http.')
    bar = re.compile(r'//*')
    pr = ["rt","@","http","https","'s",'...', 'english', 'translation','):',
          '. .', '..','2-5','<3',']:','“','”','’','. ...','___','__','=(','‘','—','°','¢','•','®','—','…',
          '... .','--->','–','»','«','£','-->','×','->','©','\n','™','¤', '˜', 'â', '\U0001f92c', '‡', '', '\x9d', 
          '\u200d', '\x81', '\x8f', '¸', '❤', '‰', '🤔', '\u2066', '\u2069', '🤷', '‚', '¬', '🤣', '‹', 'ª', '☺', '„',
         '´', '·', '🤘', '🤗', '✋', '‼', 'ºðÿ', '✊', '☑', '¶', '¥', '\x8d', '¯', '²ðÿ', '\xad', '❌', '↺', '¨', 'âƒ', 
         'ðÿž', 'ã', '・', '🤤', '🤧', 'œðÿ', 'º', '♡', '✌', '♥', '⬇', '✅', 'œ', '€', '🤢', '†', '⚫', 'ˆðÿ', 'ˆ' ,'µ', 
         '\u200b', '¾', '✝', '¼', 'ºhttp', '§', '🤓', '✖', '⚠', 'âœ', 'ðÿœ', 'ðÿœ²ðÿœ³ðÿœ', '❛', '❜', '☠', '🤸', '🤡', 
         '〝', '─', '〞', '🆗', '❣', '♠', '🤞', '🤑', '\U0001f92b', 'ツ', '⚡', '🥂', '🤕', '❀', 'ᴴᴰ', '¦', '➡', '¿', '♀',
          '🤦', 'xd', '\U0001f92e', '\u2060', '♂', '⚽', '✨', '▶', '\U0001f92a', '⚔', '―', '♾', '►', '🧀', '\U0001f970', 
         '\U0001f9d0', '☝', '❓','🆚', '\U0001f92f', '͡', '🥚', 'xk', '🤙', 'ʖ', '͜', '⤵']
    
    
    tknzr = TweetTokenizer(reduce_len=True, strip_handles=True)
    tokens = tknzr.tokenize(words.lower())
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tknzr.tokenize(words.lower())
    snowball = SnowballStemmer('spanish')
    lemmas = [snowball.stem(t) for t in tokens]
    stoplist = stopwords.words('spanish')
    lemmas_clean = [w for w in lemmas if w not in stoplist]
    punctuation = set(string.punctuation)
    lemmas_punct = [w for w in lemmas_clean if  w not in punctuation]
    lemmas_clean = [w for w in lemmas_punct if not w.startswith('@') if w not in pr 
                if not bar.search(w) if not ht.search(w) if not char.search(w) if not punct.search(w)
                if not w.isdigit() if not emoji_pattern.search(w)]
    return lemmas_clean

class LexicalStats (BaseEstimator, TransformerMixin):
    """Extract lexical features from each document"""
    
    def number_sentences(self, doc):
        sentences = sent_tokenize(doc, language='english')
        return len(sentences)

    def fit(self, x, y=None):
        return self

    def transform(self, docs):
        return [{'length': len(doc),
                 'num_sentences': self.number_sentences(doc)}
                for doc in docs]

class PosStats(BaseEstimator, TransformerMixin):
    """Obtain number of tokens with POS categories"""

    def stats(self, doc):
        tokens = english_tokenizer(doc)
        tagged = pos_tag(tokens, tagset='universal')
        counts = Counter(tag for word,tag in tagged)
        total = sum(counts.values())
        #copy tags so that we return always the same number of features
        pos_features = {'NOUN': 0, 'ADJ': 0, 'VERB': 0, 'ADV': 0, 'CONJ': 0, 
                        'ADP': 0, 'PRON':0, 'NUM': 0}
        
        pos_dic = dict((tag, float(count)/total) for tag,count in counts.items())
        for k in pos_dic:
            if k in pos_features:
                pos_features[k] = pos_dic[k]
        return pos_features
    
    def transform(self, docs, y=None):
        return [self.stats(doc) for doc in docs]
    
    def fit(self, docs, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
    
class TwitterStats(BaseEstimator, TransformerMixin):
    def extract_features(self, doc):
        features = {}
        text = pprocess_twitter.preprocess(doc)
        features['hashtags'] = text.count('<hashtag>') + text.count('<hastag>')
        features['urls'] = text.count('<url>')
        features['mentions'] = text.count('<user>')
        features['capital'] = text.count('<allcaps>')
        features['emojis'] = emoji.emoji_count(text)
        features['exclamations'] = text.count('!')
        features['repetition'] = text.count('<repeat>')
        return features
    def fit(self, docs, y=None):
        return self
    
    def transform(self, docs):
        return [self.extract_features(doc) for doc in docs]
    
def hashtag_lexicon_generation(df):
    hashtag_lexicon = set()
    tweets = df['text'].values
    for tweet in tweets:
        hashtag_lexicon = hashtag_lexicon | set([re.sub(r"(\W+)$", "", j).lower() for j in set([i for i in tweet.split() 
        if i.startswith("#")])])
    if '' in hashtag_lexicon:
        hashtag_lexicon.remove('')
    return hashtag_lexicon

class LowerTransformer(BaseEstimator, TransformerMixin):

    def transform(self, docs):
        return [doc.lower() for doc in docs]

    def fit(self, docs, y=None):
        return self
    
def subjectivity_lexicon_generation():
    subjectivity_words = set(subjectivity.words(categories='subj'))
    subjectivity_lexicon = [word for word in subjectivity_words if word not in stopwords.words('english') 
                            and word not in string.punctuation]
    objectivity_words = set(subjectivity.words(categories='obj'))
    objectivity_words = [word for word in objectivity_words if word not in stopwords.words('english') 
                           and word not in string.punctuation]
    objectivity_words = list(set(objectivity_words) - set(subjectivity_words))
    
    subjectivity_lexicon = [subjectivity_words, objectivity_words]
    
    return subjectivity_lexicon

class SubjectivityStats(BaseEstimator, TransformerMixin):
    
    def extract_features(self, doc):
        features = {}
        blobed = TextBlob(doc)
        #language = blobed.detect_language()
        #if language != 'en':
        #    blobed = blobed.translate(to='en')
        stats = blobed.sentiment
        features['polarity'] = stats.polarity
        features['subjectivity'] = stats.subjectivity
        return features
         
    def transform(self, docs):
        return [self.extract_features(doc) for doc in docs]
    
    def fit(self, docs, y=None):
        return self
    
def read_swn(objective_threshold=0.5):
    all_ = list([word for word in swn.all_senti_synsets() if word.obj_score() < objective_threshold])
    lex = pd.DataFrame.from_dict(
        {word.synset.lemmas()[0].name(): word.pos_score() - word.neg_score() for word in all_}, orient='index')
    lex.columns = ['value']
    return lex

def read_anew(path):
    lex = pd.read_csv(path)
    lex['value'] = MinMaxScaler((-1,1)).fit_transform(lex['Valence Mean'].values.reshape(-1,1))
    return lex

def read_afinn(path):
    lex = None
    with codecs.open(path, 'r', encoding='utf-8') as f:
        lex = f.readlines()
    lex = {entry.split('\t')[0]: int(entry.split('\t')[1]) for entry in lex}
    lex = pd.DataFrame.from_dict(lex, orient='index')
    lex.columns = ['value']
    return lex

        
