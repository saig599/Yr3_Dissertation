import pandas as pd
import textwrap
import gensim
from gensim.utils import simple_preprocess
from gensim import models
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag as tagging
import os
import numpy as np
np.random.seed(2018)


data = pd.read_csv('abstracts.csv', encoding= 'ISO-8859-1', error_bad_lines=False)
data_text = data[['All Abstracts']]
data_text['index'] = data_text.index
abstracts = data_text
abstracts = abstracts.dropna(subset=['All Abstracts'])

stemmer = SnowballStemmer("english")

stop_words = set(stopwords.words("english"))
lemmtizer = WordNetLemmatizer()



def lemmatize_stemming(text):
     return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
     result = []
     for token in gensim.utils.simple_preprocess(text):
          if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
               result.append(lemmatize_stemming(token))
     return result


doc_sample = abstracts[abstracts['index'] == 3].values[0][0]
# print('original document: ')
words = []
for word in doc_sample.split(' '):
     words.append(word)

# print("original abstract: ")
# print(words)
# print('\n\n tokenized and lemmatized abstract: ')
# print(preprocess(doc_sample))

processed_abstracts = abstracts['All Abstracts'].map(preprocess)
processed_abstracts[:20]

dictionary = gensim.corpora.Dictionary(processed_abstracts)
count = 0
for k, v in dictionary.iteritems():
     print(k,v, dictionary.dfs[k])
     count += 1
     if count > 15:
          break
dictionary.filter_extremes(no_below=20, no_above=0.5, keep_n=500000)

bow_corpus = [dictionary.doc2bow(abstract) for abstract in processed_abstracts]

bow_doc_3 = bow_corpus[5]
print(bow_doc_3)
for i in range(len(bow_doc_3)):
      print("Word {} (\"{}\") appears {} time.".format(bow_doc_3[i][0],
                                                      dictionary[bow_doc_3[i][0]],
                                                     bow_doc_3[i][1]))


if __name__ == '__main__':
     lda = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=100, workers=10,
                                            per_word_topics=True)
     for i in range(0, lda.num_topics-1):
          print('topic:'+'\n' + lda.print_topic(i))










