# 🤗 HuggingFace Pipelines + MLflow Experiment Tracking

A hands-on exploration of two essential AI engineering tools — HuggingFace Transformers for running pre-trained models, and MLflow for tracking and comparing ML experiments.

---

## 💡 What This Project Covers

**Part 1 — HuggingFace Pipelines**
Loading and running pre-trained models from HuggingFace Hub for different NLP tasks — sentiment analysis, question answering, and understanding the difference between extractive and generative QA.

**Part 2 — MLflow Experiment Tracking**
Logging parameters and metrics across multiple experiment runs and comparing them visually in the MLflow dashboard — solving the problem of manually tracking results in spreadsheets or memory.

---

## 🧠 Why I Built This

In my MedBERT project I was manually tracking ROUGE and BLEU scores across different training runs — different chunk sizes, learning rates, epochs — all in my head. I had no systematic way to compare which settings worked best.

MLflow solves exactly that. And HuggingFace is what made it possible to use a domain-specific model like MedBERT instead of a general model that couldn't understand medical terminology.

This project is me properly learning both tools that I had used without fully understanding in my earlier projects.

---

## 📁 Project Structure

```
pipeline/
├── sentiment.py        # HuggingFace sentiment analysis pipeline
├── qa_pipeline.py      # HuggingFace question answering pipeline
├── mlflow_demo.py      # MLflow experiment tracking demo
├── .env                # API keys (never pushed to GitHub)
├── .gitignore
└── README.md
```

---

## 🤗 Part 1 — HuggingFace Pipelines

### What is HuggingFace?
HuggingFace is an open-source platform that provides pre-trained models, datasets, and tools for NLP and AI tasks. Instead of training a model from scratch — which requires massive compute and data — you download a pre-trained model and either use it directly or fine-tune it on your own data.

### Sentiment Analysis
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love building AI projects!")
# Output: [{'label': 'POSITIVE', 'score': 0.9995}]
```

### Loading a Specific Model by Name
```python
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
```
You specify a model name when the default model isn't suitable for your domain. For medical text you'd load MedBERT, for legal text a legal-specific model — same reason my RAG pipeline failed in 2024 with a general embedding model on medical data.

### Question Answering — Extractive QA
```python
qa = pipeline("question-answering")
result = qa(
    question="What is the capital of France?",
    context="France is a country in Europe. Its capital city is Paris."
)
# Output: {'answer': 'Paris', 'score': 0.991, 'start': 51, 'end': 56}
```

### Extractive vs Generative QA
- **Extractive QA** — finds and extracts the answer span directly from the context. Returns start/end character positions. What HuggingFace QA pipeline does.
- **Generative QA** — reads the context and generates an answer in its own words. What Gemini/GPT does in our RAG pipeline.

The QA pipeline is essentially the "Generate" step of a RAG pipeline — it takes context and a question and finds the answer within that context.

---

## 📊 Part 2 — MLflow Experiment Tracking

### What is MLflow?
MLflow is an experiment tracking tool for ML projects. It automatically logs parameters, metrics, and results of every training run so you can compare them visually — no manual spreadsheets needed.

### Running the Demo
```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("my_first_experiment")

# Experiment Run 1
with mlflow.start_run(run_name="run_1"):
    mlflow.log_param("chunk_size", 500)
    mlflow.log_param("k", 3)
    mlflow.log_metric("rouge_score", 0.65)
    mlflow.log_metric("bleu_score", 0.58)

# Experiment Run 2
with mlflow.start_run(run_name="run_2"):
    mlflow.log_param("chunk_size", 1000)
    mlflow.log_param("k", 5)
    mlflow.log_metric("rouge_score", 0.78)
    mlflow.log_metric("bleu_score", 0.71)
```

### Viewing the Dashboard
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open `http://127.0.0.1:5000` → Model Training tab → click your experiment.

### Important Note
MLflow doesn't calculate your scores — YOU calculate them from your model's output and pass the numbers to MLflow. MLflow just stores, tracks, and visualises them. Like a marksheet — the teacher calculates marks, the marksheet just displays them.

### Real World Usage
In a real training loop it looks like this:
```python
for experiment in experiments:
    with mlflow.start_run():
        model = train(chunk_size=experiment["chunk_size"])
        rouge = calculate_rouge(model)
        mlflow.log_metric("rouge_score", rouge)
        # Logs automatically — no manual tracking needed
```

---

## ⚙️ Setup & Run

**1. Install dependencies**
```bash
pip install transformers torch mlflow python-dotenv
```

**2. Run HuggingFace demos**
```bash
python sentiment.py
python qa_pipeline.py
```

**3. Run MLflow demo**
```bash
python mlflow_demo.py
```

**4. Open MLflow dashboard**
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Go to `http://127.0.0.1:5000` → Model Training → my_first_experiment

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| HuggingFace Transformers | Pre-trained model pipelines |
| DistilBERT | Sentiment classification model |
| MLflow 3.13 | Experiment tracking and dashboard |
| SQLite | MLflow backend database |

---

## 🔑 Key Things I Learned

**Why domain-specific models matter**
A general HuggingFace model works fine for everyday text. But for medical, legal, or technical domains you need a model pre-trained on that domain's data — like MedBERT for medical text. This is the same reason my RAG pipeline gave poor results in 2024 — the embedding model was too general.

**Extractive vs Generative QA**
HuggingFace's QA pipeline extracts answers directly from text — fast and precise but limited to what's written. Generative models like Gemini form answers in their own words — more flexible but can hallucinate. Choosing the right one depends on your use case.

**MLflow solves a real problem**
Without MLflow, comparing 10 training runs means maintaining a manual log of every parameter and metric. MLflow automates this completely — every run is logged, timestamped, and comparable in one dashboard.

---

## 🚀 What I'd Add Next

- Connect MLflow tracking to the RAG pipeline to log retrieval quality metrics
- Fine-tune a HuggingFace model on custom data and track training with MLflow
- Use HuggingFace embeddings instead of Google embeddings in the RAG pipeline
- Deploy a HuggingFace model as a FastAPI endpoint
