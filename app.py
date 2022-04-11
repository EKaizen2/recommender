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
tfidf_headline_features = pickle.load(open('PickleModel.pkl','rb'))


def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:#002E6D;padding:20px;font-weight:15px"> 
    <h1 style ="color:white;text-align:center;">Ephraim Adongo News Article Recommender</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)
    default_value_goes_here = ""
#     uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")

    story = st.selectbox(
         'Select one article to get a recommendation of similar articles: ',
         ('‘The Voice’ Blind Auditions Make History With First Trans Contestant', '‘RuPaul’s Drag Race All Stars 3’ Episode 7 Recap: We Can Never Go Back To Before', 'Would Dr. King Take A Knee? 6 Ways His Radical Spirit Lives On Today', 'Why Would DeVos Cancel A Student Loan Rule That Works?', 'Veterans Affairs Secretary David Shulkin Ousted From White House'))
    selectedheadline = story
    if story == '‘The Voice’ Blind Auditions Make History With First Trans Contestant':
        story = 1
    elif story == "‘RuPaul’s Drag Race All Stars 3’ Episode 7 Recap: We Can Never Go Back To Before":
        story = 10
    elif story == "Would Dr. King Take A Knee? 6 Ways His Radical Spirit Lives On Today":
        story = 100
    elif story == "Why Would DeVos Cancel A Student Loan Rule That Works?":
        story = 200
    else:
        story = 500

    global dataframe
#     if uploaded_file:
    df = pd.read_excel('news_articles.xlsx')
    news_articles = df

    result = ""
    
    if st.button("Recommend"):
      def tfidf_based_model(row_index, num_similar_items):
        couple_dist = pairwise_distances(tfidf_headline_features,tfidf_headline_features[row_index])
        indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
        df = pd.DataFrame({'Publish_date': news_articles['date'][indices].values,
                   'Headline':news_articles['headline'][indices].values,
                    'Euclidean similarity with the queried article': couple_dist[indices].ravel()})
        print("="*30,"Queried article details","="*30)
        print('headline : ',news_articles['headline'][indices[0]])
        print("\n","="*25,"Recommended articles : ","="*23)

        #return df.iloc[1:,1]
        return df.iloc[1:,]
      result = tfidf_based_model(story, 11)
      st.write("========================== Queried article details ==========================")
      st.write("Selected headline :" +selectedheadline)
      st.write("=========================== Recommended articles :  =========================")
      st.write(result)
if __name__ == '__main__':
    main()
