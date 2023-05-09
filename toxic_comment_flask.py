#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import pandas as pd


# In[2]:


# set display option
pd.set_option('display.max_colwidth', 1000)


# In[3]:


# load the saved or dumped pipeline
pipeline = load("toxic_comment_classification.joblib")


# In[4]:


# Test the model again 
word=["eat shit and die"]


# In[5]:


# check prediction now
pipeline.predict(word)


# In[6]:


app=Flask(__name__)


# In[7]:


@app.route('/',methods=['GET','POST'])
def home():
    pred=None
    c=[]
    col=['severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
    if request.method == 'POST':
        user_input = request.form['comment']
        print(user_input)
        cat=pipeline.predict([user_input])
        print(cat)
        cat = pd.DataFrame(cat,columns=col)
        for i in range(len(cat)):
            if cat.columns[(cat == 1).iloc[i]].notna().all():
                c=(cat.columns[(cat == 1).iloc[i]].values)
                print(c)
    return render_template('home.html',k=c)


# In[8]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




