import streamlit as st
import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from recsysNN_utils import *

# Streamlit UI

# Configura a página principal com título, layout em tela cheia e barra lateral expandida
st.set_page_config(page_title="Sistemas de Recomendação", layout="wide")

st.title("Teste Sistemas de Recomendação")

st.image('images/movie_camera.png', width=100)

def l2_normalize(x):
    return tf.linalg.l2_normalize(x, axis=1)

keras.config.enable_unsafe_deserialization()

# Load the saved model when the app starts
@st.cache_resource
def load_model_1():
    return tf.keras.models.load_model("NN_model.keras")

@st.cache_resource
def load_model_2():
    return tf.keras.models.load_model("NNm_model.keras", custom_objects={'l2_normalize': l2_normalize})

model = load_model_1()

model_m = load_model_2()

movies_df = pd.read_csv('data/content_movie_list.csv')
movies = movies_df['title']

# Load Data, set configuration variables
item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
num_item_features = item_train.shape[1] - 1  # remove movie id at train time
uvs = 3  # user genre vector start
ivs = 3  # item genre vector start
u_s = 3  # start of columns to use in training, user
i_s = 1  # start of columns to use in training, items

# scale training data
item_train_unscaled = item_train
user_train_unscaled = user_train
y_train_unscaled    = y_train

scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)

scalerUser = StandardScaler()
scalerUser.fit(user_train)
user_train = scalerUser.transform(user_train)

scalerTarget = MinMaxScaler((-1, 1))
scalerTarget.fit(y_train.reshape(-1, 1))
y_train = scalerTarget.transform(y_train.reshape(-1, 1))
#ynorm_test = scalerTarget.transform(y_test.reshape(-1, 1))

item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test       = train_test_split(y_train,    train_size=0.80, shuffle=True, random_state=1)

def sq_dist(a,b):
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """

    n = a.shape[0]
    d = 0.0
    for l in range(n):
        d = d + (a[l]-b[l])**2
    
    return d

scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:,i_s:])

dim = len(vms)
dist = np.zeros((dim, dim))

# Compute the distance matrix
for i in range(dim):
    for j in range(dim):
        dist[i, j] = sq_dist(vms[i, :], vms[j, :])

# Mask the diagonal
m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))

st.write('Este aplicativo de teste utiliza o **Small MovieLens Latest Dataset**, que contém 100.000 avaliações e 3.600 aplicações de tags aplicadas a 9.000 filmes por 600 usuários.')
st.write('Ele ajusta um modelo de rede neural ao conjunto de dados de maneira a entender a similaridade entre diferentes títulos apenas analisando os reviews dos usuários.')

# Selectbox para escolher a opção
selected_movie = st.selectbox("Escolha um filme de sua preferência:", movies)

st.link_button(f"Acessar Dataset", "https://grouplens.org/datasets/movielens/latest/")

# Predict button
if st.button("Recomendar Filmes Similares"):
    # Specify the movie ID
    chosen_movie_id = int(movies_df[movies_df["title"] == selected_movie]['movieId'].iloc[0])

    # Find the index of the chosen movie in item_vecs
    chosen_movie_idx = np.where(item_vecs[:, 0] == chosen_movie_id)[0][0]

    # Get the top 5 most similar movies
    sorted_indices = np.argsort(m_dist[chosen_movie_idx])[:5]

    # Create a DataFrame to store results
    results = []

    for min_idx in sorted_indices:
        movie1_id = int(item_vecs[chosen_movie_idx, 0])
        movie2_id = int(item_vecs[min_idx, 0])
        results.append({
            "movie1_title": movie_dict[movie1_id]['title'],
            "movie1_genres": movie_dict[movie1_id]['genres'],
            "movie2_title": movie_dict[movie2_id]['title'],
            "movie2_genres": movie_dict[movie2_id]['genres']
        })

    results_df = pd.DataFrame(results)

    st.write("De acordo com o filme escolhido é provavel que você goste também: ")
    st.write(f"1 - {results_df['movie2_title'][0]}")
    st.write(f"2 - {results_df['movie2_title'][1]}")
    st.write(f"3 - {results_df['movie2_title'][2]}")
    st.write(f"4 - {results_df['movie2_title'][3]}")
    st.write(f"5 - {results_df['movie2_title'][4]}")