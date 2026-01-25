import os

import requests
import streamlit as st

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

API_URL = os.getenv("API_URL", "http://backend:8000")

# --- ZARZĄDZANIE SESJĄ ---
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
    st.session_state["username"] = None

# Inicjalizacja pamięci dla wyników wyszukiwania (ŻEBY NIE ZNIKAŁY)
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None
if "content_results" not in st.session_state:
    st.session_state["content_results"] = None

# ==========================================
# 1. LOGOWANIE I REJESTRACJA
# ==========================================
if st.session_state["user_id"] is None:
    st.title("🔐 Zaloguj się do Kina")

    with st.expander("🛠️ Debug Connection"):
        st.write(f"API URL: `{API_URL}`")
        if st.button("Test Ping"):
            try:
                r = requests.get(f"{API_URL}/health", timeout=1)
                st.success(f"Status: {r.status_code}")
            except Exception as e:
                st.error(f"Błąd: {e}")

    tab_login, tab_reg = st.tabs(["Logowanie", "Rejestracja"])

    with tab_login:
        username_l = st.text_input("Nick", key="login_user")
        password_l = st.text_input("Hasło", type="password", key="login_pass")
        if st.button("Zaloguj"):
            try:
                resp = requests.post(
                    f"{API_URL}/login",
                    json={"username": username_l, "password": password_l},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state["user_id"] = data["user_id"]
                    st.session_state["username"] = data["username"]
                    st.success("Zalogowano! Przeładowuję...")
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", "Błąd logowania"))
            except Exception as e:
                st.error(f"Błąd połączenia: {e}")

    with tab_reg:
        username_r = st.text_input("Wybierz Nick", key="reg_user")
        password_r = st.text_input("Wybierz Hasło", type="password", key="reg_pass")
        if st.button("Utwórz konto"):
            try:
                resp = requests.post(
                    f"{API_URL}/register",
                    json={"username": username_r, "password": password_r},
                )
                if resp.status_code == 200:
                    st.success("Konto utworzone! Możesz się zalogować.")
                else:
                    st.error(resp.json().get("detail", "Błąd rejestracji"))
            except Exception as e:
                st.error(f"Błąd: {e}")

    st.stop()

# ==========================================
# 2. FUNKCJE UI
# ==========================================


def send_rating(movie_id, rating_value):
    """Wysyła ocenę do API"""
    payload = {
        "user_id": st.session_state["user_id"],
        "movie_id": movie_id,
        "rating": rating_value,
    }
    try:
        resp = requests.post(f"{API_URL}/ratings/", json=payload)
        if resp.status_code == 200:
            # Używamy toast (dymek) I success (trwały komunikat) dla pewności
            st.toast(f"✅ Oceniono na {rating_value}!", icon="⭐")
        else:
            st.error(f"Błąd zapisu: {resp.text}")
    except Exception as e:
        st.error(f"Błąd połączenia: {e}")


def render_movie_card(movie):
    """Wyświetla pojedynczy film."""
    mid = movie.get("movie_id") or movie.get("id")
    title = movie.get("title", "Bez tytułu")
    poster = movie.get("poster")

    with st.container(border=True):
        if poster and poster.startswith("http"):
            st.image(poster, use_container_width=True)
        else:
            st.image(
                "https://via.placeholder.com/300x450?text=No+Poster",
                use_container_width=True,
            )

        st.markdown(f"**{title}**")

        # Expander do oceny
        with st.expander("⭐ Oceń"):
            rating_val = st.slider("Gwiazdki", 0.0, 5.0, 3.0, 0.5, key=f"slider_{mid}")
            # Callback przycisku nie powoduje zniknięcia listy, bo lista jest w session_state
            if st.button("Zapisz", key=f"btn_{mid}"):
                send_rating(mid, rating_val)


# ==========================================
# 3. GŁÓWNA APLIKACJA
# ==========================================

with st.sidebar:
    st.header(f"👤 {st.session_state['username']}")
    st.caption(f"ID Użytkownika: {st.session_state['user_id']}")
    if st.button("Wyloguj"):
        # Czyścimy sesję przy wylogowaniu
        st.session_state["user_id"] = None
        st.session_state["username"] = None
        st.session_state["search_results"] = None
        st.session_state["content_results"] = None
        st.rerun()

st.title(f"Popcorn dla Ciebie, {st.session_state['username']}! 🍿")

tab1, tab2, tab3 = st.tabs(["🔥 Rekomendacje", "🔎 Wyszukiwarka", "📊 Admin Dashboard"])

# --- TAB 1: REKOMENDACJE ---
with tab1:
    if st.button("Odśwież propozycje"):
        st.rerun()

    try:
        user_id = st.session_state["user_id"]
        # Tutaj nie musimy cachować w sesji, bo rekomendacje ładują się zawsze przy wejściu
        resp = requests.get(f"{API_URL}/recommend/{user_id}?top_n=5")

        if resp.status_code == 200:
            data = resp.json()
            st.info(
                f"Algorytm: {data['source']} (Model: {data.get('model_version', 'v1')})"
            )
            cols = st.columns(5)
            for idx, movie in enumerate(data["recommendations"]):
                with cols[idx % 5]:
                    render_movie_card(movie)
        else:
            st.warning("Oceń kilka filmów, aby dostać rekomendacje!")
    except Exception as e:
        st.error(f"Błąd backendu: {e}")

# --- TAB 2: WYSZUKIWARKA (Z NAPRAWIONYM ZNIKANIEM) ---
with tab2:
    search_query = st.text_input("Wpisz tytuł filmu (np. Matrix)")
    col_search, col_content = st.columns(2)

    # Przycisk 1: Szukanie SQL
    with col_search:
        if st.button("Szukaj w Bazie"):
            if search_query:
                resp = requests.get(f"{API_URL}/movies?search={search_query}&limit=5")
                if resp.status_code == 200:
                    # ZAPISUJEMY WYNIK DO SESJI
                    st.session_state["search_results"] = resp.json()
                    st.session_state["content_results"] = (
                        None  # Czyścimy drugą listę dla porządku
                    )
                else:
                    st.error("Błąd wyszukiwania")

    # Przycisk 2: Content Based
    with col_content:
        if st.button("Podobne tematycznie (AI)"):
            if search_query:
                resp = requests.get(f"{API_URL}/recommendations/content/{search_query}")
                if resp.status_code == 200:
                    data = resp.json()
                    # ZAPISUJEMY WYNIK DO SESJI
                    if "recommendations" in data:
                        st.session_state["content_results"] = data
                        st.session_state["search_results"] = (
                            None  # Czyścimy drugą listę
                        )
                    else:
                        st.info(data.get("message"))
                        st.session_state["content_results"] = None

    st.divider()

    # --- WYŚWIETLANIE WYNIKÓW Z PAMIĘCI (SESJI) ---
    # Dzięki temu, nawet jak klikniesz "Zapisz" (i strona się przeładuje),
    # kod wejdzie w te if-y i wyświetli filmy znowu.

    if st.session_state["search_results"]:
        st.caption("Wyniki z bazy danych:")
        cols = st.columns(5)
        for idx, m in enumerate(st.session_state["search_results"]):
            with cols[idx % 5]:
                render_movie_card(m)

    if st.session_state["content_results"]:
        data = st.session_state["content_results"]
        st.success(f"Filmy podobne do: {data['searched_movie']}")
        cols = st.columns(3)
        for idx, m in enumerate(data["recommendations"]):
            with cols[idx % 3]:
                render_movie_card(m)

# --- TAB 3: ADMIN ---
# --- TAB 3: ADMIN ---
with tab3:
    st.header("Panel Administratora")

    col_kpi, col_actions = st.columns(2)

    with col_actions:
        st.subheader("⚙️ Operacje MLOps")
        if st.button("🚀 Dotrenuj Model (Retrain)"):
            with st.spinner("Wysyłanie sygnału do backendu..."):
                try:
                    resp = requests.post(f"{API_URL}/admin/retrain")
                    if resp.status_code == 200:
                        st.success("Sygnał wysłany! Model uczy się w tle.")
                        st.info("Obserwuj logi w terminalu, aby zobaczyć postęp.")
                    else:
                        st.error("Błąd backendu.")
                except Exception as e:
                    st.error(f"Nie udało się połączyć: {e}")

    if st.button("Odśwież dane"):
        st.rerun()
    try:
        resp = requests.get(f"{API_URL}/admin/stats")
        if resp.status_code == 200:
            stats = resp.json()
            c1, c2 = st.columns(2)
            c1.metric("Liczba Ocen", stats.get("total_ratings"))
            c2.metric("Użytkownicy", stats.get("active_users"))
            st.subheader("Top Filmy")
            st.dataframe(stats.get("top_movies"))
    except Exception:
        st.error("Admin API nie odpowiada")
