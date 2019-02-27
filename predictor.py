import pandas as pd
import numpy as np
import itertools
import re

from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
%pylab inline

data = pd.read_csv('fake-news-detection/data.csv',encoding='ISO-8859-1')
data.head()

X_init = data['Body'].values
y = data['Label'].values
X_fin = []
print(len(X_init))
for k in X_init:
    X_fin.append(re.sub(r"\s", ' ', str(k)))
X_train, X_test, y_train, y_test = train_test_split(X_fin, y, test_size=0.33, random_state=53)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

tfidf_vectorizer.get_feature_names()[-10:]

tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
tfidf_df.head()

def predict(name):
    clf = name()
    clf.fit(tfidf_train, y_train)
    pred = clf.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.4f" % score)
    y_pred = clf.predict(tfidf_test)
    array = metrics.confusion_matrix(y_pred,y_test)
    print("precision =", np.sum(array[0][0] / (array[0][0]+array[0][1])))
    print("recall =", np.sum(array[0][0] / (array[0][0]+array[1][0])))
    df_cm = pd.DataFrame(array)
    plt.figure(figsize = (7,7))
    sns.heatmap(df_cm, annot=True)
    plt.show()
    return clf
clf1 = predict(MultinomialNB)
clf2 = predict(RandomForestClassifier)
clf3 = predict(LogisticRegression)

def prec(clf):
    y_pred = clf.predict(tfidf_test)

    average_precision = average_precision_score(y_test, y_pred)

    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.2])
    plt.xlim([0.0, 1.2])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()
    
prec(clf1)
prec(clf2)
prec(clf3)

'''a = """JetNation FanDuel League; Week 4
% of readers think this story is Fact. Add your two cents.
(Before It's News)
Our FanDuel league is back again this week. Here are the details:
$900 in total prize money. $250 to the winner. $10 to enter.
Remember this is a one week league, pick your lineup against the salary cap and next week if you want to play again you can pick a completely different lineup if you want.
Click this link to enter — http://fanduel.com/JetNation
You can discuss this with other NY Jets fans on the Jet Nation message board. Or visit of on Facebook.
Source: http://www.jetnation.com/2017/09/27/jetnation-fanduel-league-week-4/"""
tfidf_a = tfidf_vectorizer.transform([a])
y = clf.predict(tfidf_a)
print(y)#,y_test[0])
'''
