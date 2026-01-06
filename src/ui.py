import pandas as pd
import requests
import streamlit as st

import config

API_URL = "https://movie-recommender-wiktor.onrender.com"

st.set_page_config(page_title="System Rekomendacji Filmów", page_icon="🎬")


st.title("🎬 Twój Asystent Filmowy")
st.write("Wybierz ID użytkownika, a AI dobierze dla niego najlepsze filmy!")


with st.sidebar:
    st.header("⚙️ Panel Sterowania")
    user_id = st.number_input(
        "Podaj ID użytkownika:", min_value=1, max_value=100, value=1
    )
    top_n = int(
        st.number_input(
            "Podaj oczekiwaną ilość rekomendacji:",
            min_value=1,
            max_value=config.CACHE_MAX_ITEMS,
            value=3,
        )
    )

    if st.button("Sprawdź połączenie z serwerem"):
        try:
            r = requests.get(f"{API_URL}/health")
            if r.status_code == 200:
                st.success("Serwer działa poprawnie!")
            else:
                st.error(f"Błąd serwera: {r.status_code}")
        except:
            st.error("Nie można połączyć z API. Czy link jest poprawny?")

if st.button("🔍 Znajdź filmy", type="primary"):
    with st.spinner("AI analizuje oceny..."):
        try:
            response = requests.get(f"{API_URL}/recommend/{user_id}")

            if response.status_code == 200:
                data = response.json()
                recs = data.get("recommendations", [])

                if recs:
                    st.subheader(f"🍿 Filmy wybrane dla Ciebie (User {user_id}):")

                    # Tworzymy siatkę (grid) z plakatami
                    cols = st.columns(len(recs))  # Tyle kolumn ile filmów

                    for idx, movie in enumerate(recs):
                        with cols[idx]:
                            # Jeśli jest plakat, wyświetl go
                            if movie.get("poster"):
                                st.image(movie["poster"], use_container_width=True)
                            else:
                                # Placeholder jeśli brak zdjęcia
                                st.image(
                                    "https://via.placeholder.com/300x450?text=No+Poster",
                                    use_container_width=True,
                                )

                            # Tytuł pod zdjęciem (zmniejszony font)
                            st.caption(f"**{movie['title']}**")
                else:
                    st.warning("Brak rekomendacji.")

        except Exception as e:
            st.error(f"Wystąpił błąd połączenia: {e}")


st.divider()
st.caption(f"Backend obsługiwany przez: {API_URL}")
