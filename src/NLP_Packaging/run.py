import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from processing import tf_idf_scores
from data import t0, t1, t2,t3,t4,t5,t6
nlp = spacy.load("en_core_web_sm")
l = [t0, t1, t2,t3,t4,t5,t6]
df=tf_idf_scores(l)
sns.set()
fig,ax=plt.subplots(figsize=(15,3))
sns.heatmap(df,ax=ax)
plt.show()
