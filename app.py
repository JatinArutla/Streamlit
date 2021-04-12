import streamlit as st
import pandas as pd
import numpy as np

header = st.beta_container()
pre_processing = st.beta_container()

with header:
    st.title('Netflix TV show recommendation system')
    netflix_overall = pd.read_csv("netflix_titles.csv")
    netflix_shows = netflix_overall[netflix_overall['type']=='TV Show']
    netflix_shows = netflix_shows.copy()
    arr = netflix_shows['title']
    arr = arr.tolist()

    sel_col, disp_col = st.beta_columns(2)
    options = sel_col.selectbox('Select a TV show (Just overwrite the title in the search box)', options=arr)

with pre_processing:

    netflix_shows['duration'], netflix_shows['season'] = netflix_shows['duration'].str.split(' ', 1).str
    netflix_shows.drop(columns=['season','rating'], axis=1, inplace=True)

    def clean_data(x):
        return str.lower(x.replace(" ", ""))
    filledna = netflix_shows.fillna('')
    filledna.head(2)
    filledna['release_year'] = filledna['release_year'].apply(str)
    features = ['title', 'cast', 'description', 'listed_in', 'duration', 'release_year', 'country']
    filledna = filledna[features]

    for feature in features:
        filledna[feature] = filledna[feature].apply(clean_data)
    def create_soup(x):
        return x['title'] + ' ' + x['cast'] + ' ' +x['listed_in']+' '+ x['description']+' '+ x['duration']+' '+ x['country']
    filledna['soup'] = filledna.apply(create_soup, axis=1)


    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(filledna['soup'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    filledna.reset_index(drop=True, inplace=True)
    indices = pd.Series(filledna.index, index=filledna['title'])

    def get_recommendations_new(title, cosine_sim):
        title = title.replace(' ','').lower()
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        shows_indices = [i[0] for i in sim_scores]
        arr = netflix_shows['title'].iloc[shows_indices]
        df = arr.to_frame()
        df.index = np.arange(len(df))
        return df

    result = get_recommendations_new(options, cosine_sim2)
    st.write(result)
    st.text('Happy watching')