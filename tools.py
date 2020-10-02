import re
import nltk
from unidecode import unidecode
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

my_stop_word_list = get_stop_words('french')
s_w=list(set(my_stop_word_list))
s_w=[elem.lower() for elem in s_w]
fr = SnowballStemmer('french')

#construction du corpus 
Corpus=pd.read_csv('corpus.csv')
Corpus['label']=Corpus['rating']
Corpus['l_review']=Corpus['review'].apply(lambda x:len(x.split(' ')))
Corpus=Corpus[Corpus['l_review']>5]

positif=Corpus[Corpus['label']>3].sample(391)
negatif=Corpus[Corpus['label']<3]

Corpus_new=pd.concat([positif,negatif],ignore_index=True)[['review','label']]

for ind in Corpus_new['label'].index:
    if Corpus_new.loc[ind,'label'] > 3:
        Corpus_new.loc[ind,'label']=1
    elif Corpus_new.loc[ind,'label'] < 3:
        Corpus_new.loc[ind,'label']=0

print(len(Corpus_new))


def nettoyage(string):
    l=[]
    string=unidecode(string.lower())

    string=" ".join(re.findall("[a-zA-Z]+", string))
    
    for word in string.split():
        if word in s_w:
            continue
        else:
            l.append(fr.stem(word))
    return ' '.join(l)


def model(phrase):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
    user = transformer.fit_transform(loaded_vec.fit_transform([nettoyage(phrase)]))

    cls=pickle.load(open("cls.pkl", "rb"))
    if (cls.predict(user)[0])== 1 :
        texte="possitif"
    else:
        texte="négatif"
    #return (cls.predict(user)[0],cls.predict_proba(user).max())
    return (texte, round(cls.predict_proba(user).max(),3))

def entrainement():
    #print (Corpus['review_net'][:2])
    Corpus_new['review_net']=Corpus_new['review'].apply(nettoyage)
    #print (Corpus['review_net'][:2])

    vectorizer = TfidfVectorizer()
    vectorizer.fit(Corpus_new['review_net'])
    X=vectorizer.transform(Corpus_new['review_net'])
    

    #Save vectorizer.vocabulary_
    pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))

    y=Corpus_new['label']
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.15)
    cls=LogisticRegression(max_iter=300).fit(x_train,y_train)
   
    #Save classifier
    pickle.dump(cls,open("cls.pkl","wb"))

    return (round(cls.score(x_val,y_val),3))
    #return (y)