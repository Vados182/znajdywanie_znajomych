import json
import streamlit as st
import pandas as pd  # type: ignore
import joblib
import plotly.express as px  # type: ignore

DATA = 'welcome_survey_simple_v1.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v1.json'

# Definicja poprawnej kolejności dla wykresów
CATEGORY_ORDERS = {
    "age": ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'],
    "edu_level": ['Podstawowe', 'Średnie', 'Wyższe']
}

@st.cache_resource
def get_model():
    # 1. Ładujemy oryginalny pipeline za pomocą joblib
    raw_pipeline = joblib.load('welcome_survey_clustering_pipeline_v1.pkl')
    
    # 2. Jeśli to obiekt PyCareta, wyciągamy z jego kroków czysty model Scikit-learn (np. KMeans),
    # aby uniknąć jakichkolwiek błędów importu czy brakujących metod w chmurze.
    if hasattr(raw_pipeline, 'steps'):
        return raw_pipeline.steps[-1][1]
    return raw_pipeline

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def get_all_participants():
    model_pipeline = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    
    # Generujemy czyste numery klastrów
    cluster_labels = model_pipeline.predict(all_df)
    
    df_with_clusters = all_df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    return df_with_clusters

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

# Predykcja dla nowego użytkownika przy użyciu czystego modelu
predicted_cluster_labels = model_pipeline.predict(person_df)

# Dopasowanie do formatu kluczy w Twoim pliku JSON (np. "Cluster 7")
predicted_cluster_id = f"Cluster {predicted_cluster_labels[0]}"

predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

# --- GŁÓWNY PANEL ---
st.header(f"Najbliżej Ci do grupy: {predicted_cluster_data['name']}")
st.markdown(f"*{predicted_cluster_data['description']}*")

# Filtrowanie tabeli po czystym numerze klastra (w postaci tekstowej cyfry, np. "7")
same_cluster_df = all_df[all_df["Cluster"].astype(str) == str(predicted_cluster_labels[0])]
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