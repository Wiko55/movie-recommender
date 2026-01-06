import requests
import streamlit as st

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
API_URL = "http://localhost:8000"

st.title("🎬 Twój Asystent Filmowy")

# Zakładki
tab1, tab2 = st.tabs(["👤 Rekomendacje dla Usera", "🔎 Podobne do Filmu"])

# --- ZAKŁADKA 1: USER ---
with tab1:
    st.write("Wybierz ID użytkownika, a AI dobierze dla niego filmy.")
    col1, col2 = st.columns(2)
    with col1:
        user_id = st.number_input(
            "Podaj ID użytkownika:", min_value=1, max_value=200000, value=1
        )
    with col2:
        top_n_user = st.slider("Ile filmów?", 1, 10, 5, key="slider_user")

    if st.button("Znajdź dla Usera"):
        with st.spinner("Szukam..."):
            try:
                res = requests.get(
                    f"{API_URL}/recommend/{user_id}", params={"top_n": top_n_user}
                )
                if res.status_code == 200:
                    data = res.json()
                    recs = data.get("recommendations", [])

                    # Wyświetlanie (funkcja pomocnicza by się przydała, ale skopiujmy pętlę)
                    if recs:
                        cols = st.columns(len(recs))
                        for idx, movie in enumerate(recs):
                            with cols[idx]:
                                if isinstance(movie, dict):
                                    poster = movie.get("poster")
                                    if poster:
                                        st.image(poster, use_container_width=True)
                                    else:
                                        st.image(
                                            "https://via.placeholder.com/300x450",
                                            use_container_width=True,
                                        )
                                    st.caption(f"**{movie.get('title')}**")
                                else:
                                    st.write(movie)
                    else:
                        st.warning("Brak wyników.")
            except Exception as e:
                st.error(f"Błąd: {e}")

# --- ZAKŁADKA 2: FILM ---
with tab2:
    st.write("Wpisz tytuł filmu, który lubisz (np. Toy Story, Batman, Inception).")
    col1, col2 = st.columns(2)
    with col1:
        movie_query = st.text_input("Tytuł filmu:")
    with col2:
        top_n_movie = st.slider("Ile filmów?", 1, 10, 5, key="slider_movie")

    if st.button("Znajdź podobne"):
        if not movie_query:
            st.warning("Wpisz tytuł!")
        else:
            with st.spinner(f"Szukam filmów podobnych do '{movie_query}'..."):
                try:
                    # Request do nowego endpointu
                    res = requests.get(
                        f"{API_URL}/similar/{movie_query}",
                        params={"top_n": top_n_movie},
                    )
                    if res.status_code == 200:
                        data = res.json()
                        recs = data.get("recommendations", [])

                        if recs:
                            st.subheader(f"Ponieważ lubisz '{movie_query}':")
                            cols = st.columns(len(recs))
                            for idx, movie in enumerate(recs):
                                with cols[idx]:
                                    if isinstance(movie, dict):
                                        poster = movie.get("poster")
                                        if poster:
                                            st.image(poster, use_container_width=True)
                                        else:
                                            st.image(
                                                "https://via.placeholder.com/300x450",
                                                use_container_width=True,
                                            )
                                        st.caption(f"**{movie.get('title')}**")
                        else:
                            st.warning("Nie znalazłem takiego filmu w bazie :(")
                    else:
                        st.error("Błąd API.")
                except Exception as e:
                    st.error(f"Błąd: {e}")
