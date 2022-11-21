from operator import index
import streamlit as st
import plotly.express as px
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
#stemming class from nltk
from nltk.stem.porter import PorterStemmer
#count vectorizor
from sklearn.feature_extraction.text import CountVectorizer
#cosine similarties to calculate the similarty measure between movies
from sklearn.metrics.pairwise import cosine_similarity


def recommend(movie):
    #movie_index = new_df[new_df['title'] == movie].index[0]
    movie_list = data[data['tagging'].str.contains(movie)]
    if len(movie_list):  
        movie_idx= movie_list.index[0]
        distances = similarity[movie_idx]
        movies_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6]
    
        #
        print('Recommendations for {0} :\n'.format(movie_list.iloc[0]['tagging']))
        for i in movies_list:
            print(data.iloc[i[0]].tagging)
    else:
        return "No movies found. Please check your input"

#defining the helper stemming function
def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)



data = pd.read_csv('mnc.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("MNC DATASET")
    choice = st.radio("Navigation", ["EDA","Modelling"])

# if choice == "Upload":
#     # st.title("Upload Your Dataset")
    
#     if file: 
#         data = pd.read_csv(file, index_col=None)
#         data.to_csv('mnc.csv', index=None)
#         st.dataframe(data)

if choice == "EDA": 
    # st.title("Exploratory Data Analysis")
    # profile_data = data.profile_report()
    # st_profile_report(profile_data)
    pass

if choice == "Modelling": 
    ps = PorterStemmer()

    cv = CountVectorizer(max_features=5000,stop_words='english')
    vectors = cv.fit_transform(data['tagging']).toarray()

    #devine method cosin similarity
    similarity = cosine_similarity(vectors)

    txt = st.text_area('Enter text: ')
    st.write (str(recommend(txt)))