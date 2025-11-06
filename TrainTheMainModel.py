#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import re

df = pd.read_csv("balady_google_reviews.csv", encoding="utf-8-sig")  


def clean_text(text):
    text = str(text)
    emoji_pattern = re.compile(
        "["                       
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U0001F900-\U0001F9FF"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)

    
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"\d+", "", text)

    
    text = re.sub(r'(.)\1{2,}', r'\1', text)

   
    text = re.sub(r'\s+', ' ', text).strip()

    return text


df["content_cleaned"] = df["content"].apply(clean_text)


df_nonempty = df[df["content_cleaned"].str.strip() != ""].copy()


df_final = df_nonempty[["content_cleaned", "score"]]
df_final.to_csv("cleaned_with_score.csv", index=False, encoding="utf-8-sig")

print("✅ تم تنظيف النصوص وحفظ العمود مع score في 'cleaned_with_score.csv'")


# In[29]:


import emoji
import pandas as pd
import re


def clean_text(text):
    text = str(text)
    
  
    text = emoji.replace_emoji(text, replace='')
    
   
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"\d+", "", text)

   
    text = re.sub(r'(.)\1{2,}', r'\1', text)

  
    text = re.sub(r'\s+', ' ', text).strip()

    return text


df["content_cleaned"] = df["content"].apply(clean_text)


df_nonempty = df[df["content_cleaned"].str.strip() != ""].copy()


df_final = df_nonempty[["content_cleaned", "score"]]
df_final.to_csv("cleaned_with_score.csv", index=False, encoding="utf-8-sig")

print("✅ تم إزالة كل الإيموجي وحفظ العمود مع score بنجاح")


# In[32]:


df["content_cleaned"].isna()


# In[33]:


df["content_cleaned"].info()


# In[35]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

.csv", encoding="utf-8-sig")
df = df[df["content_cleaned"].str.strip() != ""].copy()

X = df["content_cleaned"]
y = df["score"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,3), max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])


model.fit(X_train, y_train)
print("✅ تم تدريب النموذج بنجاح")


y_pred = model.predict(X_test)
print("📊 تقرير التصنيف:")
print(classification_report(y_test, y_pred))

print("📊 مصفوفة الالتباس:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(model, "sentiment_model_pipeline_balanced.pkl")
print("✅ تم حفظ النموذج المحسّن في 'sentiment_model_pipeline_balanced.pkl'")


# In[ ]:




