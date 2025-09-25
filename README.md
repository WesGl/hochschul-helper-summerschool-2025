# HKA Hochschul-Helper (Summer School 2025)

Ein agentenbasiertes Assistenzsystem für Studierende der Hochschule Karlsruhe. Dieses Projekt bietet eine intelligente Plattform, die Studierenden hilft, alle relevanten Informationen rund um das Studium schnell und zuverlässig zu finden.

## Features

- **Stundenplan-Retrieval**: Automatische Extraktion und Suche von Vorlesungs-, Übungs- und Raumplänen aus den offiziellen HKA-Stundenplänen.
- **RAG (Retrieval-Augmented Generation)**: KI-gestützte Beantwortung von Fragen auf Basis der aktuellen Hochschul-Dokumente und Stundenpläne.
- **Quellenangabe & Konfidenz**: Jede Antwort enthält die verwendeten Quellen und eine Vertrauensbewertung.
- **Kalender-Export**: Exportiere Termine als ICS-Datei oder direkt in Google Calendar.
- **Agentic Workflow**: Flexible Tool-Auswahl und Planung durch einen intelligenten Agenten.
- **Chainlit UI**: Moderne Chat-Oberfläche für die Interaktion mit dem System.

## Projektstruktur

- `src/` – Hauptlogik, Agent, Modelle, Tools
- `scripts/` – Daten-Ingestion und Crawler für Stundenpläne
- `data/` – Gespeicherte Daten für RAG
- `vectordb/` – Persistente Vektordatenbank (Chroma)

## Installation & Setup

1. **Python 3.10+ installieren**
2. Abhängigkeiten installieren:
    ```bash
    uv sync
    ```
3. **Playwright installieren** (für den Crawler):
    ```bash
    uv run playwright install
    ```
4. **Stundenpläne crawlen & indizieren:**
    ```bash
    uv run scripts/time_table_crawler.py
    uv run scripts/ingest_timetables.py
    ```
5. **Chainlit-UI starten:**
    ```bash
    uv run chainlit run src/app.py -w
    ```

## Nutzung

Stelle Fragen zu Stundenplänen, Vorlesungen, Räumen oder anderen hochschulbezogenen Themen direkt im Chat. Der Agent sucht die passenden Informationen und liefert präzise Antworten mit Quellenangabe.

## Erweiterung

- Neue Tools können in `src/tools/` hinzugefügt werden.
- Die Agent-Logik ist modular und kann für weitere Hochschulservices erweitert werden.
