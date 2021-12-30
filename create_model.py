import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import warnings

import random

#Model
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score 

from sklearn.metrics import confusion_matrix,classification_report

import dill

warnings.filterwarnings("ignore")

dataset=pd.read_csv("data.csv",error_bad_lines=False)

dataset.head()

dataset.shape

dataset["strength"].unique()

dataset.info()

dataset.dropna(inplace=True)

plt.figure(figsize=(8,8))
sns.countplot(dataset.strength)

password_=np.array(dataset)

random.shuffle(password_)

password_

X=[passwords[0] for passwords in password_]
y=[passwords[1] for passwords in password_]


def make_chars(inputs):
    characters=[]
    for letter in inputs:
        characters.append(letter)
    return characters

vectorizer=TfidfVectorizer(tokenizer=make_chars)

X_=vectorizer.fit_transform(X)

X_.shape

vectorizer.get_feature_names()
X_[0]

first_=X_[0].T.todense()

vec=pd.DataFrame(first_,index=vectorizer.get_feature_names(),columns=['tfidf'])

vec

vec.sort_values(by=['tfidf'],ascending=False)
x_train,x_test,y_train,y_test=train_test_split(X_,y,test_size=0.27,random_state=42)

x_train.shape,x_test.shape

classifier=[]
# classifier.append(LogisticRegression(multi_class='ovr',n_jobs=-1))
# classifier.append(LogisticRegression(multi_class='multinomial',solver='newton-cg',n_jobs=-1))
classifier.append(xgb.XGBClassifier(n_jobs=-1))
# classifier.append(MultinomialNB())

# result=[]
# for model in classifier:
#     a=model.fit(x_train,y_train)
#     result.append(a.score(x_test,y_test))

# result1=pd.DataFrame({'score':result,
#                       'algorithms':['logistic_regr_ovr',
#                                     'logistic_regr_mutinomial',
#                                     'xgboost','naive bayes']})

# result1

xgb_classifier=xgb.XGBClassifier(n_jobs=-1)

xgb_classifier.fit(x_train,y_train)

pred=xgb_classifier.predict(x_test)

confusion_matrix(y_test,pred)

print(classification_report(y_test,pred))


# ########################
model_file=open("xgb_classifier.pkl","wb")
dill.dump(xgb_classifier,model_file)
model_file.close()

dill.dump(vectorizer, open("vectorizer.pkl", "wb"))

password="abc123@ABC"
password=vectorizer.transform([password])
xgb_classifier.predict(password)
a=xgb_classifier.predict_proba(password)
a
a[0][1]
