FROM python:3.9-slim

WORKDIR /app

# Instalacija sistemskih zavisnosti
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Kopiranje requirements fajla
COPY requirements.txt .

# Instalacija Python zavisnosti
RUN pip install --no-cache-dir -r requirements.txt

# Kopiranje svih fajlova projekta
COPY . .

# Pokretanje skripte
CMD ["python", "scripts/main.py"]