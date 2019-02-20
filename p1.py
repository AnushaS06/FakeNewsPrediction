from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv('train.csv',encoding='ISO-8859-1')
data['X_labels'] = data['Label'].map({False: 0, True: 1})

obj = TfidfVectorizer()
X = obj.fit_transform(data['Statement'])
Y = data['X_labels'].values

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

model = LogisticRegression()
model.fit(Xtrain,Ytrain)
print("Classification rate for LR:" , model.score(Xtest, Ytest))


yPred = model.predict(Xtest)
array = confusion_matrix(yPred,Ytest)
df_cm = pd.DataFrame(array)
plt.figure(figsize = (2,2))
sns.heatmap(df_cm, annot=True)
plt.show()
