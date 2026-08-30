# Docker — Porownanie ANFIS z klasycznymi modelami ML

Kontener uruchamia dashboard Streamlit sluzacy do porownywania modelu ANFIS
z klasycznymi algorytmami uczenia maszynowego.

## Wymagania

- Docker Engine 24+ z wtyczka `docker compose`

## Uruchomienie

Wszystkie polecenia wykonuj z katalogu `.tools/docker`:

```bash
cd .tools/docker
docker compose up --build
```

Po starcie aplikacja jest dostepna pod adresem <http://localhost:8501>.

Katalog \`results/\` jest zamontowany z hosta, wiec wygenerowane wykresy
i metryki zostaja na dysku po zatrzymaniu kontenera.

## Zatrzymanie

```bash
docker compose down
```

Aby usunac takze wolumeny z danymi:

```bash
docker compose down -v
```
