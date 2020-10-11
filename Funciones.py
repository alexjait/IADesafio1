from bs4 import BeautifulSoup
from itertools import chain
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.models import Sequential
from keras.preprocessing import text, sequence
from nltk import pos_tag
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import WordPunctTokenizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from string import punctuation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud,STOPWORDS
import gensim
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re,string,unicodedata
import seaborn as sns
import tensorflow as tf

#Todo en lower
def to_lower(texto):
    lower_text = texto.lower()
    return lower_text

#Html parser
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Borrar corchetes
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Borrar urls
def remove_urls(text):
    return re.sub(r'http\S+', '', text)

# Borrar stopwords
def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)    
    
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

def denoise_text(text):
    text = to_lower(text)
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_urls(text)
    text = remove_stopwords(text)
    text = text.strip()
    return text

def agregarResultado(resultados, modelo, train_acc, test_acc):
    resultados_dict = { 
        'Modelo': modelo, 
        'Set': 'Train',
        'Accuracy': train_acc}
    df_temp = pd.DataFrame([resultados_dict], columns=resultados_dict.keys())
    resultados = pd.concat([resultados, df_temp], axis=0)

    resultados_dict = { 
        'Modelo': modelo, 
        'Set': 'Test',
        'Accuracy': test_acc}
    df_temp = pd.DataFrame([resultados_dict], columns=resultados_dict.keys())
    resultados = pd.concat([resultados, df_temp], axis=0)
    return resultados

def ejecutarModelo1(nombre_modelo, grid, vectorizer, train, test, y_train, y_test, resultados, tfidfTransformer=None):
    X_train = vectorizer.fit_transform(train)
    
    if tfidfTransformer is None:
        grid.fit(X_train, y_train)
    else:
        grid.fit(tfidfTransformer.fit_transform(X_train), y_train)
        
    print("Best cross-validation score: {:.4f}".format(grid.best_score_));
    print("Best parameters: ", grid.best_params_);
    X_test = vectorizer.transform(test);
    model = grid.best_estimator_;
    y_pred_train = model.predict(X_train);
    y_pred_test = model.predict(X_test);  
    return agregarResultado(
        resultados,
        nombre_modelo, 
        accuracy_score(y_train, y_pred_train),
        accuracy_score(y_test, y_pred_test))    

def ejecutarModelo2(nombre_modelo, model, vectorizer, X_train, X_test, y_train, y_test, resultados):
    model.fit(X_train, y_train)    
    
    return agregarResultado(
                resultados,
                nombre_modelo, 
                model.score(X_train, y_train),
                model.score(X_test, y_test)), pd.DataFrame({
                'atributo': vectorizer.get_feature_names(), 
                'importancia': model.feature_importances_}).sort_values('importancia', ascending = False).head(10)  

def graficoBalanceado():
    plt.figure(figsize=(8, 8))
    sns.set(style="darkgrid")
    ax = sns.countplot(x="label", data=df)
    ax.set(xticklabels=['fake', 'real'])
    plt.title("Fake news vs real news");
    plt.savefig(fname='Graficos/01_Balanceado.png')

def graficarPie(x, labels, autopct, nombre_archivo):
    pie, ax = plt.subplots(figsize=[8,8])
    labels = labels
    plt.pie(x=x, 
            autopct=autopct, labels=labels);
    pie.savefig(nombre_archivo)
    