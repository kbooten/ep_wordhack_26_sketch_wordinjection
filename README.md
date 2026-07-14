# Live Narration

A Flask app that transcribes audience speech locally and uses OpenAI to generate an evolving narration incorporating words and phrases from the users.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key to `.env`:

```
OPENAI_API_KEY=your_key_here
```

## Run

```
python app.py
``