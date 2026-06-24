import streamlit as st
import pandas as pd  # type: ignore
import numpy as np
import plotly.express as px  # type: ignore

DATA = 'welcome_survey_simple_v1.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v1.json'

CATEGORY_ORDERS = {
    "age": ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'],
    "edu_level": ['Podstawowe', 'Średnie', 'Wyższe']
}

# Słownik odwzorowujący unikalne kategorie (uporządkowane dokładnie tak, jak ułożył je oryginalny enkoder)
COLUMNS_MAPPING = {
    "age": ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'],
    "edu_level": ['Podstawowe', 'Średnie', 'Wyższe'],
    "fav_animals": ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'],
    "fav_place": ['Nad wodą', 'W lesie', 'W górach', 'Inne'],
    "gender": ['Mężczyzna', 'Kobieta']
}

# Twoja oryginalna macierz z modelu (wymiar: 8 klastrów x 21 cech)
CENTROIDS = np.array([
    [-1.734723475976807e-18, 0.18750000000000003, 0.4375, 0.3125, 6.938893903907228e-18, 6.938893903907228e-18, 0.0625, -3.469446951953614e-18, -1.734723475976807e-18, 0.0625, 0.9375000000000001, 0.0, 1.0, 2.7755575615628914e-17, 2.7755575615628914e-17, -6.938893903907228e-18, -1.6653345369377348e-16, 0.9999999999999999, 1.1102230246251565e-16, -1.3877787807814457e-17, 0.6875],
    [-1.734723475976807e-18, 0.03333333333333342, 5.551115123125783e-17, 0.7, 0.0, 0.03333333333333334, 0.23333333333333345, -3.469446951953614e-18, -1.734723475976807e-18, 0.06666666666666665, 0.9333333333333333, 0.0, 0.8333333333333333, 0.0, 0.13333333333333336, 0.033333333333333326, 0.9666666666666669, -1.1102230246251565e-16, 5.551115123125783e-17, 0.03333333333333332, 0.7666666666666667],
    [-8.673617379884035e-19, 0.36842105263157887, 0.21052631578947373, 0.15789473684210525, 0.2631578947368421, 6.938893903907228e-18, 0.0, -1.734723475976807e-18, -8.673617379884035e-19, 0.9999999999999999, -2.220446049250313e-16, 0.052631578947368446, 0.4736842105263158, 0.2105263157894737, 0.26315789473684215, -3.469446951953614e-18, 0.1578947368421052, 0.10526315789473686, 0.6842105263157896, 0.05263157894736841, 1.0],
    [-1.734723475976807e-18, 5.551115123125783e-17, 5.551115123125783e-17, 1.0000000000000002, 6.938893903907228e-18, 6.938893903907228e-18, 0.0, -3.469446951953614e-18, -1.734723475976807e-18, -2.7755575615628914e-17, 1.0, 0.11764705882352942, 0.23529411764705888, 0.4117647058823529, 0.1764705882352941, 0.058823529411764705, -1.6653345369377348e-16, -5.551115123125783e-17, 0.9411764705882353, 0.0588235294117647, 0.8235294117647058],
    [-1.734723475976807e-18, 0.43750000000000006, 0.25, -1.1102230246251565e-16, 0.062499999999999986, 0.1875, 0.0625, -3.469446951953614e-18, -1.734723475976807e-18, -2.7755575615628914e-17, 1.0, 0.375, 0.37500000000000006, 0.1875, 0.06250000000000003, -6.938893903907228e-18, -1.6653345369377348e-16, 0.0625, 0.9375, -1.3877787807814457e-17, 0.6875],
    [-1.734723475976807e-18, 0.11764705882352941, 0.8235294117647058, -1.1102230246251565e-16, 6.938893903907228e-18, 6.938893903907228e-18, 0.0, 0.05882352941176472, -1.734723475976807e-18, 0.11764705882352941, 0.8823529411764706, 0.0, 0.8823529411764706, 2.7755575615628914e-17, 2.7755575615628914e-17, 0.11764705882352945, 0.8823529411764703, -5.551115123125783e-17, 1.1102230246251565e-16, 0.11764705882352941, 0.4705882352941177],
    [-1.734723475976807e-18, 0.09090909090909091, 0.9090909090909092, 0.0, -6.938893903907228e-18, 6.938893903907228e-18, 0.0, -3.469446951953614e-18, -1.734723475976807e-18, 0.2727272727272727, 0.7272727272727273, 0.0, 0.0, 0.2727272727272727, 0.7272727272727272, -6.938893903907228e-18, 0.18181818181818177, 0.5454545454545454, 5.551115123125783e-17, 0.27272727272727276, 0.9090909090909092],
    [0.07142857142857145, 0.07142857142857145, 0.07142857142857151, 0.6428571428571429, 6.938893903907228e-18, 0.07142857142857144, 0.0, 0.07142857142857145, 0.07142857142857145, 0.14285714285714285, 0.7857142857142857, 0.7142857142857141, 1.1102230246251565e-16, 0.28571428571428575, 2.7755575615628914e-17, -3.469446951953614e-18, 0.857142857142857, 0.14285714285714288, 1.1102230246251565e-16, -6.938893903907228e-18, 0.8571428571428571]
])

def build_features_vector(row):
    """Tworzy wektor binarny dopasowany długością (21) bezpośrednio do centroidów"""
    vector = []
    for col, values in COLUMNS_MAPPING.items():
        current_val = str(row[col]).strip()
        for val in values:
            if current_val == str(val).strip():
                vector.append(1.0)
            else:
                vector.append(0.0)
                
    features_array = np.array(vector)
    # Wymuszamy, by wektor miał dokładnie 21 elementów (tyle ile kolumn ma model)
    if len(features_array) != CENTROIDS.shape[1]:
        features_array = np.resize(features_array, CENTROIDS.shape[1])
    return features_array

def predict_closest_cluster(row, centroids_matrix):
    """Liczy euklidesową odległość i zwraca indeks klastra (0 do 7)"""
    features_vector = build_features_vector(row)
    distances = [np.linalg.norm(features_vector - c) for c in centroids_matrix]
    return int(np.argmin(distances))

@st.cache_data
def get_cluster_names_and_descriptions():
    import json
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    clusters = []
    for i in range(len(all_df)):
        row = all_df.iloc[i]
        clusters.append(predict_closest_cluster(row, CENTROIDS))
    all_df['Cluster'] = clusters
    return all_df

# --- INICJALIZACJA DANYCH ---
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    
    age = st.selectbox("Wiek", CATEGORY_ORDERS["age"])
    edu_level = st.selectbox("Wykształcenie", CATEGORY_ORDERS["edu_level"])
    fav_animals = st.selectbox("Ulubione zwierzęta", COLUMNS_MAPPING["fav_animals"])
    fav_place = st.selectbox("Ulubione miejsce", COLUMNS_MAPPING["fav_place"])
    gender = st.radio("Płeć", COLUMNS_MAPPING["gender"])

    person_row = {
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
    }

# Poprawna predykcja indeksu klastra (zwraca liczbę int z zakresu 0-7)
predicted_cluster_idx = predict_closest_cluster(person_row, CENTROIDS)

# Konwersja na dokładny klucz z Twojego pliku JSON (np. "Cluster 0", "Cluster 1")
predicted_cluster_id = f"Cluster {predicted_cluster_idx}"
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

# --- PANEL GŁÓWNY ---
st.header(f"Najbliżej Ci do grupy: {predicted_cluster_data['name']}")
st.markdown(f"*{predicted_cluster_data['description']}*")

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_idx]
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