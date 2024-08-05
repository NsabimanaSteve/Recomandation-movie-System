import pandas as pd
import streamlit as st
import joblib

# Function to get the index of a movie title
def get_movie_title_index(movie_title, indices):
    try:
        return indices[movie_title]
    except KeyError:
        return None

# Load the content-based recommender file
with open('c:\\Users\\user\\Desktop\\AI Final Project\\content_based_recommender.pkl', 'rb') as recommend:
    loaded_model_components = joblib.load(recommend)

with open('c:\\Users\\user\\Desktop\\AI Final Project\\Merged_data.pkl', 'rb') as sd:
    Merged_data = joblib.load(sd)

loaded_vectorizer = loaded_model_components['tfidf_vectorizer']
loaded_similarity = loaded_model_components['cosine_similarity_matrix']
loaded_indices = loaded_model_components['indices_mapping']

# Function to recommend movie titles
def recommend_movie_titles(movie_title, indices, cosine_similarity, Merged_data):
    title_index = get_movie_title_index(movie_title, indices)
    if title_index is None:
        return []
    
    index = indices[movie_title]
    similarity = sorted(list(enumerate(cosine_similarity[index])), key=lambda x: x[1], reverse=True)
    similarity_list = similarity[1:7]  # Exclude the input movie itself and then recommend the most similar six
    
    recommendations = []
    for i in similarity_list:
        recommended_title = Merged_data['Title'].iloc[i[0]]
        recommended_date = Merged_data['Release Date'].iloc[i[0]]
        recommended_genre = Merged_data['Genre'].iloc[i[0]]
        recommended_poster = Merged_data['Image'].iloc[i[0]]  # Ensure these are valid URLs or paths
        recommendation_info = {
            'Poster': recommended_poster,
            'Title': recommended_title,
            'Date': recommended_date,
            'Genre': recommended_genre
        }
        recommendations.append(recommendation_info)
    
    return recommendations

def display_recommendations(movie_title, streaming_movies):
    recommendations = recommend_movie_titles(movie_title, loaded_indices, loaded_similarity, streaming_movies)
    
    st.subheader("Recommended Movies:")
    for recommendation in recommendations:
        if recommendation['Poster']:
            st.image(recommendation['Poster'], caption=recommendation['Title'], use_column_width=False, width=100)
        else:
            st.write(f"No poster available for {recommendation['Title']}")
        st.write(f"*Title:* {recommendation['Title']}")
        st.write(f"*Date:* {recommendation['Date']}")
        st.write(f"*Genre:* {recommendation['Genre']}")
        st.write("---")

def main(streaming_movies):
    st.title("D & S Movie Recommendation System") # D for Delice & S for Steve
    
    for _, row in streaming_movies.iterrows():
        if row['Poster']:
            st.image(row['Poster'], caption=row['Title'], use_column_width=True)
            
        button_key = f"recommend_{row['Title'].replace(' ', '_')}"
        if st.button(f"Get Recommendations for {row['Title']}", key=button_key):
            display_recommendations(row['Title'], streaming_movies)

if __name__ == '__main__':
    movie_data = pd.read_csv('c:\\Users\\user\\Desktop\\AI Final Project\\Movie_data.csv')
    
    if 'Image' not in movie_data.columns:
        st.error("CSV file does not contain 'Image' column.")
    else:
        main(movie_data)