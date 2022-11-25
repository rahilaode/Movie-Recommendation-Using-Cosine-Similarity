from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib



input_movie = input('Movie Name : ')

#connection = create_engine("postgresql://postgres:432000@localhost:5432/learning").connect()
#df = pd.read_sql_table('movies2', connection)
#df = df.fillna(value=np.nan)
df = pd.read_csv('movies.csv')
list_movie = df['title'].tolist()

features_selection = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in features_selection:
    df[feature] = df[feature].fillna('')
    

features_combination = df[features_selection[0]]+' '+df[features_selection[1]]+' '+df[features_selection[2]]+' '+df[features_selection[3]]+' '+df[features_selection[4]]

feature_vector = TfidfVectorizer(analyzer='word', stop_words='english').fit_transform(features_combination)
similarity = cosine_similarity(feature_vector)
find_close_match = difflib.get_close_matches(input_movie, list_movie)
close_match = find_close_match[0]
movie_index = df[df['title'] == close_match]['index'].values[0]
similarity_score = list(enumerate(similarity[movie_index]))
sorted_similar_movie = sorted(similarity_score, key = lambda x:x[1], reverse = True)

print("Movie search result : {}\n".format(input_movie))

i=1
for movie in sorted_similar_movie:
    index = movie[0]
    title = df[df.index == index]['title'].values[0]
    if(i<=20):
        print('{}. {}'.format(i, title))
        i+=1

