import pandas as pd
import requests
import streamlit as st

API_URL = "https://movie-recommender-wiktor.onrender.com"

st.set_page_config(page_title="System Rekomendacji Filmów", page_icon="🎬")


st.title("🎬 Twój Asystent Filmowy")
st.write("Wybierz ID użytkownika, a AI dobierze dla niego najlepsze filmy!")


with st.sidebar:
    st.header("⚙️ Panel Sterowania")
    user_id = st.number_input(
        "Podaj ID użytkownika:", min_value=1, max_value=1000, value=10
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
    with st.spinner("AI analizuje miliony (no, setki) ocen..."):
        try:
            response = requests.get(f"{API_URL}/recommend/{user_id}")

            if response.status_code == 200:
                data = response.json()
                recommendations = data.get("recommendations", [])

                if recommendations:
                    st.subheader(f"Filmy polecane dla Użytkownika {user_id}:")

                    for i, movie in enumerate(recommendations, 1):
                        st.info(f"🎥 {i}. {movie}")
                else:
                    st.warning(
                        "Brak rekomendacji dla tego użytkownika (może to nowy użytkownik?)."
                    )
            else:
                st.error("Wystąpił błąd po stronie serwera API.")
                st.write(response.json())

        except Exception as e:
            st.error(f"Wystąpił błąd połączenia: {e}")


st.divider()
st.caption(f"Backend obsługiwany przez: {API_URL}")
