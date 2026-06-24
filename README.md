# 👥 Znajdywanie Znajomych - Aplikacja do Segmentacji Użytkowników

Aplikacja webowa stworzona w frameworku **Streamlit**, która pozwala na interaktywną analizę i segmentację użytkowników na podstawie danych z ankiet powitalnych. Projekt wykorzystuje algorytmy uczenia maszynowego (Clustering) do automatycznego grupowania osób o podobnych profilach i zainteresowaniach.

🚀 **[KLIKNIJ TUTAJ, ABY ZOBACZYĆ APLIKACJĘ NA ŻYWO](TUTAJ_WKLEJ_LINK_Z_STREAMLIT_CLOUD)**

---

## 📊 O Projekcie & Cel Biznesowy
Głównym celem projektu było stworzenie narzędzia wspomagającego proces łączenia użytkowników w grupy (np. społecznościowe lub networkingowe). 

Aplikacja:
* Wczytuje i przetwarza dane z ankiet (`welcome_survey_simple_v1.csv`).
* Wykorzystuje wytrenowany pipeline machine learningowy (`welcome_survey_clustering_pipeline_v1.pkl`) do przypisywania nowych użytkowników do odpowiednich klastrów.
* Mapuje techniczne identyfikatory klastrów na czytelne biznesowo nazwy i opisy zapisane w pliku konfiguracyjnym JSON.

## 🛠️ Użyte Technologie i Biblioteki
* **Python** (główny język projektu)
* **Streamlit** (interfejs użytkownika i wdrożenie w chmurze)
* **Scikit-learn** (pipeline clusteringowy i uczenie maszynowe)
* **Pandas / NumPy** (manipulacja i czyszczenie danych)

## 📁 Struktura Projektu
* `app.py` - główny plik aplikacji Streamlit zarządzający interfejsem.
* `welcome_survey_clustering_pipeline_v1.pkl` - zapisany model/pipeline ML odpowiedzialny za segmentację.
* `welcome_survey_cluster_names_and_descriptions_v1.json` - słownik mapujący klastry na czytelne opisy grup.
* `welcome_survey_simple_v1.csv` - zbiór danych wejściowych z ankiety.
* `requirements.txt` - lista zależności niezbędna do uruchomienia aplikacji w chmurze.

---

