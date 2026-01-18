FROM python:3.11-slim

WORKDIR /app

# 1) Installer les deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copier l'API
COPY main.py .

# 3) Copier le modèle (depuis la racine du projet)
COPY ../model.pkl ./model.pkl

# 4) Exposer le port API
EXPOSE 8001

# 5) Lancer FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

