import requests
import streamlit as st

# Konfiguracja strony
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

API_URL = "http://localhost:8000/recommend"

# Nagłówek
st.title("🎬 Twój Asystent Filmowy")
st.write("Wybierz ID użytkownika i liczbę filmów, a AI zrobi resztę!")

# --- NOWOŚĆ: Układ kolumnowy dla inputów ---
col1, col2 = st.columns(2)

with col1:
    user_id = st.number_input(
        "Podaj ID użytkownika:", min_value=1, max_value=200000, value=1
    )

with col2:
    # Suwak do wyboru liczby rekomendacji (od 1 do 10)
    top_n = st.slider("Ile rekomendacji pokazać?", min_value=1, max_value=10, value=5)

if st.button("🔍 Znajdź filmy"):
    with st.spinner("AI analizuje Twój gust..."):
        try:
            # Pytamy API (przekazujemy dynamiczne top_n z suwaka)
            response = requests.get(f"{API_URL}/{user_id}", params={"top_n": top_n})

            if response.status_code == 200:
                data = response.json()
                recs = data.get("recommendations", [])
                source = data.get("source", "unknown")

                # Info o źródle
                if source == "cache":
                    st.info("⚡ Wynik z pamięci podręcznej (Cache)")
                elif source == "popularity_fallback":
                    st.warning(
                        "❄️ User nieznany - wyświetlamy Globalne Hity (Cold Start)"
                    )
                else:
                    st.success("🧠 Rekomendacja z modelu ML")

                if recs:
                    st.subheader(f"🍿 Filmy wybrane dla Ciebie (User {user_id}):")

                    # Tworzymy tyle kolumn, ile filmów przyszło z API
                    cols = st.columns(len(recs))

                    for idx, movie in enumerate(recs):
                        with cols[idx]:
                            # Logika Defensywna (działa dla napisów i obrazków)
                            if isinstance(movie, dict):
                                title = movie.get("title", "Bez tytułu")
                                poster = movie.get("poster")

                                if poster:
                                    st.image(poster, use_container_width=True)
                                else:
                                    st.image(
                                        "https://via.placeholder.com/300x450?text=No+Poster",
                                        use_container_width=True,
                                    )
                                st.caption(f"**{title}**")

                            elif isinstance(movie, str):
                                st.image(
                                    "https://via.placeholder.com/300x450?text=No+Poster",
                                    use_container_width=True,
                                )
                                st.caption(f"**{movie}**")
                            else:
                                st.error("Błąd danych")

                else:
                    st.warning("Brak rekomendacji.")
            else:
                st.error(f"Błąd API: {response.status_code}")

        except Exception as e:
            st.error(f"Wystąpił błąd połączenia: {e}")
