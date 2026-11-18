"""Configurazione centralizzata del backend.

I valori possono essere sovrascritti con variabili in .env
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_temperature: float = 0.3

    # RAG
    ollama_embed_model: str = "bge-m3"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "travel_kb"
    kb_dir: str = "../knowledge_base"
    rag_top_k: int = 4
    rag_score_margin: float = 0.1

    # Meteo (Open-Meteo: gratuito, nessuna chiave API)
    weather_geocoding_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    weather_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    weather_timeout: float = 8.0
    
    # Voli (database SQLite, creato e popolato all'avvio)
    flights_db_path: str = "data/flights.db"
    flights_timezone: str = "Europe/Rome"
    flights_days_ahead: int = 60          # giorni di voli generati a partire da oggi
    flights_min_coverage_days: int = 14   # sotto questa soglia i voli vengono rigenerati
    
    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # CORS: origini del frontend ammesse a chiamare l'API in sviluppo
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")


settings = Settings()
