
# coding: utf-8

# In[39]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv("mushrooms.csv")

df.head()


# In[6]:


#Do one-hit encoding
onehit = pd.get_dummies(df)
onehit.head()


# In[9]:


#Get train and test
train, test = train_test_split(onehit, test_size=0.33, random_state=14)
train.head()


# In[17]:


features = train.columns.tolist()
features.remove('class_e')
features.remove('class_p')
print(features)


# In[33]:


X_train = train[features]
X_test = test[features]
Y_train = train['class_e']
Y_test = test['class_e']

print(X_train.head)


# In[36]:


xgb = xgboost.XGBClassifier(n_jobs=-1, n_estimators=20).fit(X_train,Y_train)


# In[37]:


probs = xgb.predict_proba(X_test)


# In[44]:


preds = probs[:,1]
print(preds)


# In[52]:


fpr, tpr, threshold = sklearn.metrics.roc_curve(Y_test, preds)
roc_auc = sklearn.metrics.auc(fpr, tpr)
print(roc_auc)


# In[51]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

