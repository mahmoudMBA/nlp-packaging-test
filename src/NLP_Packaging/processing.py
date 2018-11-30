import spacy
import pandas as pd
nlp = spacy.load("en_core_web_sm")
def lemmatize(d):
  a=[token.lemma_ for token in d if not token.is_punct and not token.is_space and token.lower_ not in STOP_WORDS]
  return a

def jaccard_similarity(list1, list2):
    a=set(lemmatize(nlp(list1)))
    b=set(lemmatize(nlp(list2)))
    intersection = len((a).intersection(b))
    #print(list(a.intersection(b)))
    union = (len(a) + len(b)) - intersection
    return float(intersection / union)

def t(text,l):
  i=0
  for t in l :
    if t==text:
       i=i+1
  return i


def tf(t,doc):
  
  a=[token.lemma_ for token in doc if not token.is_punct and not token.is_space and token.lower_ not in STOP_WORDS]
  
  
  return a.count(''.join(lemmatize(nlp(t))))


def tf(t):
  t=nlp(t)
  t=t[0].lemma_
  return t


def tf(t,doc):
  t=nlp(t)
  i=0
  for token in doc:
    if token.lemma_ == t[0].lemma_:
      i+=1
  return i



def idf(t,l):
  i=0
  for d in l:
    if tf(t,nlp(d)) >=1:
      i+=1
    else :
      i=i
  if i>0 :
    return 1/i
  else:
    return 0

def tf_idf(t,doc,D):
  return tf(t,doc)*idf(t,D)



def all_lemmas(D):

 
  for doc in D:
    
    a=[token.lemma_ for token in nlp(doc) if not token.is_punct and not token.is_space and token.lower_ not in STOP_WORDS]
  return set(a)



def all_lemmas(D):
  a=[]
  for doc in D:
    for token in nlp(doc):
       if not token.is_punct and not token.is_space and token.lower_ not in STOP_WORDS:
          a.append(token.lemma_)
  return set(a)


def tf_idf_doc(doc,D):
  dic={}
  lemmas= all_lemmas(D)
  for lemma in lemmas:
    dic[lemma]= tf_idf(lemma,doc,D)
  return dic



def tf_idf_doc(doc,D):
  dic={}
  #lemmas= all_lemmas(D)
  for token in doc:
    dic[token.lemma_]= tf_idf(token.lemma_,doc,D)
  return dic



def tf_idf_scores(D):
  col=[]
  lemmas = all_lemmas
  for lemma in lemmas:
    row=[]
    for doc in D:
      row.append(doc)
    col.append(row)
  df=pd.DataFrame(col)
  return df

def tf_idf_scores(D):
  l=[]
  lis=[ f't{i}' for i in range(len(D))]
  for doc in D:
    l.append( tf_idf_doc(nlp(doc),D))
  df=pd.DataFrame(data=l,index=lis)
  return df