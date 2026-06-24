import json
import streamlit as st
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
from pycaret.clustering import load_model, predict_model

DATA = 'welcome_survey_simple_v1.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v1.json'

# Definicja poprawnej kolejności dla wykresów
CATEGORY_ORDERS = {
    "age": ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'],
    "edu_level": ['Podstawowe', 'Średnie', 'Wyższe']
}

@st.cache_resource
def get_model():
    # PyCaret ładuje plik bez podawania rozszerzenia .pkl
    return load_model('welcome_survey_clustering_pipeline_v1')

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def get_all_participants():
    model_pipeline = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    
    # PyCaret dedykowany sposób na predykcję dla całej tabeli
    predictions = predict_model(model_pipeline, data=all_df)
    
    # PyCaret dopisuje wynik do kolumny 'Cluster'
    all_df['Cluster'] = predictions['Cluster']
    return all_df

# --- LOGIKA APLIKACJI ---
model_pipeline = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    
    age = st.selectbox("Wiek", CATEGORY_ORDERS["age"])
    edu_level = st.selectbox("Wykształcenie", CATEGORY_ORDERS["edu_level"])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
    }])

# Predykcja dla nowego użytkownika przez PyCaret
predicted_df = predict_model(model_pipeline, data=person_df)

# Ponieważ w Twoim JSON klucze to np. "Cluster 7", upewniamy się, że format pasuje
raw_cluster_id = predicted_df['Cluster'].iloc[0]
predicted_cluster_id = str(raw_cluster_id)

if not predicted_cluster_id.startswith("Cluster"):
    predicted_cluster_id = f"Cluster {predicted_cluster_id}"

predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

# --- GŁÓWNY PANEL ---
st.header(f"Najbliżej Ci do grupy: {predicted_cluster_data['name']}")
st.markdown(f"*{predicted_cluster_data['description']}*")

# Filtrowanie tabeli
same_cluster_df = all_df[all_df["Cluster"].astype(str) == str(raw_cluster_id)]
st.metric("Liczba osób w Twojej grupie", len(same_cluster_df))

st.write("---")
st.header("Statystyki Twojej grupy")

charts_config = [
    {"col": "age", "title": "Rozkład wieku", "label": "Wiek"},
    {"col": "edu_level", "title": "Rozkład wykształcenia", "label": "Wykształcenie"},
    {"col": "fav_animals", "title": "Ulubione zwierzęta", "label": "Zwierzęta"},
    {"col": "fav_place", "title": "Ulubione miejsca", "label": "Miejsce"},
    {"col": "gender", "title": "Rozkład płci", "label": "Płeć"},
]

tabs = st.tabs([cfg["label"] for cfg in charts_config])

for tab, cfg in zip(tabs, charts_config):
    with tab:
        fig = px.histogram(
            same_cluster_df, 
            x=cfg["col"],
            category_orders=CATEGORY_ORDERS,
            color_discrete_sequence=['#0083B0']
        )
        fig.update_layout(
            title=cfg["title"],
            xaxis_title=cfg["label"],
            yaxis_title="Liczba osób",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)