# Scalable ML Recommendation Microservice

A containerized, full-stack recommendation engine built with an API-first approach. This microservice architecture implements both collaborative and content-based filtering, optimized for performance through caching and asynchronous operations.

## Main page with recommended movies
<img width="1422" height="805" alt="image" src="https://github.com/user-attachments/assets/07f7b7a1-9a0d-4a3b-b01f-c5389ca0c7e5" />

## Similar movies for a given title (Shrek 2 in this case)
<img width="1440" height="812" alt="image" src="https://github.com/user-attachments/assets/bc4f0710-5002-4828-a053-24f0a6364505" />


## 🎯 Architecture & Engineering Focus

This project was designed with **MLOps and Cloud/FinOps principles** in mind. Instead of a monolithic script, the system is separated into distinct layers (API, Data, Cache, Frontend) to simulate a production-ready environment.

* **Compute Optimization :** Integrated **Redis caching** to drastically reduce redundant database I/O and expensive on-the-fly matrix computations for frequent queries.
* **Asynchronous Operations:** Implemented background model retraining (`/admin/retrain`) to ensure the main FastAPI thread remains unblocked and highly available for user requests.
* **Isolated Environments:** Fully dockerized deployment (App, DB, Cache, Monitoring) ensuring parity across development and production environments.

## Core ML Features

* **Collaborative Filtering (k-NN):** Recommends items based on user-item interaction similarity.
* **Content-Based Filtering (TF-IDF):** Leverages natural language processing on item metadata to recommend items with high cosine similarity.
* **Secure Authentication:** JWT/bcrypt-based user authentication for secure endpoint access.

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend** | FastAPI, SQLAlchemy|
| **Frontend** | Streamlit |
| **Database** | PostgreSQL |
| **Caching Layer** | Redis |
| **Machine Learning** | Scikit-learn, Pandas, SciPy |
| **DevOps / MLOps** | Docker, Docker Compose |

## 🚀 Quick Start (One-Click Deployment)

The entire infrastructure is orchestrated via Docker Compose. You do not need to install local Python dependencies to run the service.

```bash
docker-compose up --build
```


[![CI Pipeline](https://github.com/Wiko55/movie-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/Wiko55/movie-recommender/actions/workflows/ci.yml)
