# %%writefile app.py%
import streamlit as st
import pickle
import openpyxl
import xlrd
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# loading the trained model
tfidf_headline_features = pickle.load(open('Pickle.pkl','rb'))


def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:#FF0000;padding:10px;font-weight:10px"> 
    <h1 style ="color:white;>Ephraim Adongo New Article Recommender</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)
    default_value_goes_here = ""
#     uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")

    global dataframe
#     if uploaded_file:
    df = pd.read_excel('news_articles.xlsx')
    news_articles = df

    result = ""
    
    if st.button("Predict"):
      def tfidf_based_model(row_index, num_similar_items):
        couple_dist = pairwise_distances(tfidf_headline_features,tfidf_headline_features[row_index])
        indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
        df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
                   'headline':news_articles['headline'][indices].values,
                    'Euclidean similarity with the queried article': couple_dist[indices].ravel()})
        print("="*30,"Queried article details","="*30)
        print('headline : ',news_articles['headline'][indices[0]])
        print("\n","="*25,"Recommended articles : ","="*23)

        #return df.iloc[1:,1]
        return df.iloc[1:,]
      result = tfidf_based_model(300, 11)
      st.write(result)

if __name__ == '__main__':
    main()
