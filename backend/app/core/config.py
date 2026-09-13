import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "DataPulse AI"
    API_V1_STR: str = "/api/v1"
    
    # Security (auth stub)
    SECRET_KEY: str = Field(default="SUPER_SECRET_DATAPULSE_KEY_FOR_JWT_TOKEN", env="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./data/datapulse.db", env="DATABASE_URL")
    
    # Ollama Narrative Layer Configuration
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field(default="mistral", env="OLLAMA_MODEL")
    
    # Simulation Settings
    SIMULATION_INTERVAL_SECONDS: int = Field(default=5, env="SIMULATION_INTERVAL_SECONDS")
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()

# Ensure data directory exists
os.makedirs("./data", exist_ok=True)
