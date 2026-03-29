# Data Collection & Exploration

Official courseware and technical framework for the **Data Collection & Exploration** module.  
This project is a live-deployed platform on **Vercel**, designed to teach the end-to-end data lifecycle.

## Mission

Transform raw, fragmented urban data into a high-integrity **Data Lake** and actionable insights.

Students learn how to:

- build automated ingestion pipelines (Python scraping and image resizing),
- design the **Transform** layer by preserving original signals and creating enriched features,
- apply **AI Vision** on images and **NLP** on reviews to generate analytical scores,
- perform robust data sanity auditing,
- run advanced exploratory data analysis (EDA),
- structure evidence for data-driven urban policy recommendations.

## Project Goal

The goal is to guide students from raw multimodal sources (tabular data, reviews, images) to defensible analytical outputs.
The pedagogical flow mirrors real-world data engineering and analytics practices used in city-scale investigations.

In this module, students analyze the socio-economic impact of short-term rentals in Paris and produce evidence-based arguments for urban decision-making.

## Deployment

This repository powers the official course support website and is intended for live use through Vercel deployment.

## Phase 2 — Student engineering deliverable (ETL)

For the **Data Collection & Exploration** module, Phase 2 ends with a **Git repository** (typically GitHub) that demonstrates a complete **Extract → Transform → Load** path:

- **Structure**: `data/raw` (Bronze), `data/processed` (Silver CSVs such as `filtered_elysee.csv`, `transformed_elysee.csv`), `scripts` (`0.4_extract.py`, `0.5_transform.py`, `0.6_load.py`, plus Phase 1 ingestion scripts as needed), `.gitignore`, and **no committed secrets** (use `.env.example` only).
- **README**: must explain prerequisites, execution order, environment variables, and expected outputs.
- **Evidence**: the README should include a **screenshot** showing **PostgreSQL** with the target database and the loaded table (e.g. `elysee_listings_silver`), e.g. from pgAdmin, DBeaver, or `psql`. The image can be stored under something like `docs/screenshots/` and referenced in Markdown.

Full wording and checklist are in the course page `pages/04-etl.html` (section *Livrable de la Phase 2*).

