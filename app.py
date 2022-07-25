import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import preprocessing 
import preprocessing
import streamlit as slt
import plotly.express as px
import plotly.graph_objects as go
import tensorflow
from tensorflow.keras.models import load_model



slt.set_page_config(layout="wide")
slt.title("Covid Fake News Prediction")
text = slt.text_area('Article Text')


slt.sidebar.header("Covid Fake News Classification")

# model1 = load_model('LSTM_RNN.h5')
# slt.write(model1)
cv = joblib.load('count_vectorizer.pkl')
mode = slt.sidebar.radio("Select any Mode",("Machine Learning","Deep Learning"))
#machine_learning = slt.sidebar.checkbox("Machine Learning")
#Deep_Learning = slt.sidebar.checkbox("Deep Learning")


if mode == "Machine Learning": 
    option = slt.sidebar.selectbox(
     'Select a model',
     ('Logestic_Regression', 'NaiveBayes', 'RandomForest','GradientBoosting'))    
    model = joblib.load(option+".pkl")
    slt.write(model)
    if slt.button("Analyze"):
        with slt.spinner("ZAINX.."):
            sentences,data = preprocessing.tokenizer(text)
            corpus = preprocessing.preprocessing(sentences)
            count_vectorized_feature = preprocessing.count_vectorizer(corpus,cv)
            prediction,data = preprocessing.prediction(count_vectorized_feature,data,model)
        
            slt.dataframe(data.style.applymap(preprocessing.color_Fake_red, subset=['Class']))

            fig = go.Figure(go.Pie(labels = ['Fake','Real'],values = data['Class'].value_counts(),hoverinfo = "label+percent",textinfo = "value") )
            slt.header("Pie chart")
            slt.plotly_chart(fig)
            slt.title("Bar chart")
            val_count  = data['Class'].value_counts()
            fig1 = plt.figure(figsize=(10,5))
            sns.barplot(val_count.index, val_count.values, alpha=0.8)
            
            slt.pyplot(fig1)


if mode == "Deep Learning": 
    option = slt.sidebar.selectbox(
     'Select a model',
     ('LSTM_RNN','GRU_RNN','Bidirectional_LSTM'))
    model = load_model(option+".h5")  

    if slt.button("Analyze"):
        with slt.spinner("ZAINX.."):
            sentences,data = preprocessing.tokenizer(text)
            corpus = preprocessing.preprocessing(sentences)
            X = preprocessing.deep_preprocessing(corpus)
            prediction = preprocessing .deep_prediction(X,data,model)
            dataset = preprocessing.pred_result(prediction,data)
            slt.dataframe(dataset.style.applymap(preprocessing.color_Fake_red, subset=['Class']))

            fig = go.Figure(go.Pie(labels = ['Fake','Real'],values = dataset['Class'].value_counts(),hoverinfo = "label+percent",textinfo = "value") )
            slt.header("Pie chart")
            slt.plotly_chart(fig)
            
            # fig1 = dataset['Class'].value_counts(sort=False).plot.bar(color='green')
            # slt.pyplot(fig1)
            #chart_data = pd.DataFrame(dataset,columns='Class')
            # df1 = dataset['Class'].value_counts().rename_axis('unique_values').reset_index(name='counts')

            # slt.bar_chart(df1)
            slt.title("Bar chart")
            val_count  = dataset['Class'].value_counts()
            fig1 = plt.figure(figsize=(10,5))
            sns.barplot(val_count.index, val_count.values, alpha=0.8)
            
            slt.pyplot(fig1)

        
        




