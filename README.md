# 🍛 SwaadStack AI

### Intelligent Real-Time Meal Completion Recommendation Engine

> _Transforming checkout into a personalized culinary consultation by stacking the right flavors at the right moment._

---

Author: Ramharsh Sanjay Dandekar

---

## 📋 Table of Contents

- [Executive Summary](#executive-summary)
- [Architecture Overview](#architecture-overview)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Detailed Module Guide](#detailed-module-guide)
- [API Reference](#api-reference)
- [Model Architecture](#model-architecture)
- [Data Pipeline](#data-pipeline)
- [Evaluation & Metrics](#evaluation--metrics)
- [Deployment](#deployment)
- [Business Impact](#business-impact)

---

## 🎯 Executive Summary

### The Problem

Current recommendation systems in food delivery suffer from **"Context Blindness"** — they fail to recognize incomplete meal patterns (e.g., a main course missing a beverage) or adapt to real-time cart changes. This results in:

- Irrelevant suggestions
- Low acceptance rates
- "Recommendation fatigue" that degrades user experience
- Suboptimal cart values (missed revenue)

### Our Solution

A **Hybrid Two-Tower + Sequential Transformer** architecture that:

1. **Understands meal flow**: Learns that Starter → Main → Side → Beverage is a natural progression
2. **Zero-shot cold-start**: Uses LLM embeddings for new items with no interaction history
3. **Real-time context**: Incorporates time-of-day, location, and user preferences
4. **Diverse recommendations**: MMR re-ranking prevents "5 types of Coke"

### Key Differentiators

| Feature            | Traditional Systems         | SwaadStack AI                                      |
| ------------------ | --------------------------- | -------------------------------------------------- |
| Cart Understanding | Static (no order awareness) | **Sequential Transformer** (understands meal flow) |
| New Items          | Requires historical data    | **Zero-shot via Sentence-BERT** embeddings         |
| Diversity          | Popularity-biased           | **MMR re-ranking** for equitable exposure          |
| Latency            | Variable                    | **< 300ms** guaranteed budget                      |
| Context            | None                        | **Temporal + Geographic** cyclical encoding        |

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                         │
│                    (< 300ms total)                            │
│                                                              │
│  ┌───────────┐   ┌──────────────┐   ┌──────────┐   ┌──────┐│
│  │  Feature   │→ │ Transformer  │→ │  FAISS   │→ │ MMR  ││
│  │  Fetch     │   │ Encoding     │   │ Search   │   │Rerank││
│  │ (< 15ms)  │   │ (< 80ms)    │   │(< 25ms) │   │(<20ms)││
│  └───────────┘   └──────────────┘   └──────────┘   └──────┘│
└──────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐
│   User Tower     │    │   Item Tower     │
│  (Transformer)   │    │  (MLP Projector) │
│                  │    │                  │
│  Cart Sequence   │    │  Item Embedding  │
│  + Context       │    │  (Sentence-BERT) │
│  (B, S, 384)     │    │  (N, 384)        │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│  User Intent     │    │  Item Latent     │
│  (B, 128)        │    │  (N, 128)        │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └──────────┬────────────┘
                    │
              Dot Product → FAISS ANN → MMR → Top-N
```

---

## 📁 Directory Structure

```
swaad-stack-ai/
├── 📄 config.py              # Centralized configuration (all hyperparameters)
├── 📄 utils.py               # Shared utilities (MMR, encoding, logging)
├── 📄 data_generator.py      # Synthetic data generation pipeline
├── 📄 model.py               # Two-Tower + Transformer architecture (PyTorch)
├── 📄 train.py               # Training pipeline with temporal split
├── 📄 feature_store.py       # Redis-based online feature store
├── 📄 inference_engine.py    # FAISS indexing + MMR re-ranking engine
├── 📄 app.py                 # FastAPI serving application
├── 📄 test_swaadstack.py     # Comprehensive test suite (35+ tests)
├── 📄 requirements.txt       # Python dependencies
├── 📄 Dockerfile             # Multi-stage production Docker image
├── 📄 docker-compose.yml     # Service orchestration (API + Redis)
├── 📄 .env                   # Environment variables
├── 📄 .gitignore             # Git ignore rules
├── 📄 README.md              # This file
├── 📁 data/                  # Generated data artifacts
│   ├── menu.json             # Menu items + embeddings
│   ├── sessions.csv          # Training sessions
│   ├── item_embeddings.npy   # Raw embedding matrix
│   └── id_mapping.json       # Item ID ↔ index mapping
├── 📁 models/                # Trained model artifacts
│   ├── swaadstack_model.pth  # Model weights
│   └── projected_item_embeddings.npy  # FAISS-ready embeddings
└── 📁 logs/                  # Application logs
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda
- (Optional) Docker & Docker Compose
- (Optional) Redis for feature store

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python data_generator.py --sessions 5000 --users 500
```

This creates:

- `data/menu.json` — 50 menu items with Sentence-BERT embeddings
- `data/sessions.csv` — 5000+ training samples following meal logic
- `data/item_embeddings.npy` — Embedding matrix

### 3. Train the Model

```bash
python train.py
```

This runs:

- Temporal train/val/test split (70/15/15)
- 15 epochs with early stopping
- Saves model to `models/swaadstack_model.pth`
- Exports FAISS-ready embeddings

### 4. Start the API

```bash
python app.py
```

The API will be available at: **http://localhost:8000**

### 5. Test Recommendations

```bash
# Get recommendations for a cart with Butter Chicken
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"cart_items": ["MAIN_001"]}'

# Browse menu
curl http://localhost:8000/menu

# Health check
curl http://localhost:8000/health
```

### 6. Run Tests

```bash
pytest test_swaadstack.py -v
```

### 7. Docker Deployment

```bash
# Build and start all services
docker compose up --build

# With Redis Commander UI (dev mode)
docker compose --profile dev up --build
```

---

## 📖 Detailed Module Guide

### `config.py` — Centralized Configuration

All hyperparameters in one place:

- **DataConfig**: Menu size, session count, embedding dimensions, geohash list
- **ModelConfig**: Transformer layers/heads, embedding sizes, latent space dim
- **TrainingConfig**: Epochs, learning rate, negative sampling, early stopping
- **InferenceConfig**: FAISS top-K, MMR lambda, latency budgets
- **FeatureStoreConfig**: Redis connection, TTLs, key prefixes
- **APIConfig**: Host, port, workers

### `data_generator.py` — Synthetic Data Pipeline

**50 realistic menu items** across 4 categories with:

- Rich text descriptions for Sentence-BERT embedding
- Cuisine types (North Indian, South Indian, Chinese, Italian, etc.)
- Dietary tags (Vegetarian, Non-Veg, Vegan, Keto, etc.)
- Realistic price points

**Session simulation** with:

- **Meal flow logic**: Main → Side → Beverage → Dessert
- **Pairing rules**: Indian curry → Naan (not Fries), Pizza → Garlic Bread
- **User profiles**: Budget, cuisine affinity, dietary preferences
- **Conditional context**: Time-of-day, geohash location
- **Realistic hour distribution**: Peaks at lunch (12-2pm) and dinner (7-10pm)

### `model.py` — Two-Tower Architecture

- **ItemTower**: 384 → 256 → 128 MLP with LayerNorm + GELU + L2 normalize
- **SequentialUserTower**: TransformerEncoder (2 layers, 4 heads) with [CLS] token
- **ContextEncoder**: Temporal sin/cos + geohash embedding → MLP
- **PositionalEncoding**: Sinusoidal encoding for meal sequence order
- **SwaadStackModel**: Full training wrapper with BCEWithLogitsLoss
- **SwaadStackInference**: Optimized no-grad wrapper for production

### `train.py` — Training Pipeline

- **CartCompletionDataset**: Negative sampling (50% hard, 50% easy)
- **Temporal split**: Chronological to prevent data leakage
- **AdamW** optimizer with **OneCycleLR** (cosine annealing + warmup)
- **Gradient clipping** (max norm = 1.0)
- **Early stopping** (patience = 5)
- **IR Metrics**: NDCG@K, Precision@K, Recall@K, AUC

### `inference_engine.py` — Real-Time Engine

- **FAISS IndexFlatIP**: Inner product search on L2-normalized vectors
- **MMR re-ranking**: Balances relevance vs. diversity
- **Cold-start fallback**: Contextual popularity by geohash + mealtime
- **Missing category detection**: "Your cart needs a Beverage!"
- **Numpy fallback**: Works even without FAISS installed

### `feature_store.py` — Redis Feature Store

- **User vectors**: Batch-updated (24h TTL), includes RFM scores
- **Cart context**: Real-time updates (session TTL)
- **Popularity cache**: Pre-computed rankings for cold-start
- **Pipeline reads**: MGET for < 15ms multi-key fetch
- **Graceful fallback**: In-memory store when Redis unavailable

### `app.py` — FastAPI Application

- **POST /recommend**: Main recommendation endpoint
- **POST /predict**: Backward-compatible alias
- **GET /menu**: Browse items with category/cuisine filters
- **GET /menu/{id}**: Single item details
- **GET /health**: Component health status
- **Middleware**: Request timing header (X-Response-Time)
- **CORS**: Configured for mobile/web clients

---

## 📡 API Reference

### POST /recommend

```json
// Request
{
  "cart_items": ["MAIN_001", "SIDE_001"],
  "user_id": "user_0001",          // optional
  "geohash": "tdr1y",              // optional
  "hour": 13,                       // optional (0-23)
  "day_of_week": 3,                 // optional (0-6)
  "top_n": 5,                       // optional (1-20)
  "diversity": 0.7                   // optional (0.0-1.0)
}

// Response
{
  "recommendations": [
    {
      "item_id": "BEV_003",
      "name": "Masala Chaas",
      "category": "Beverage",
      "price": 70,
      "cuisine": "North Indian",
      "mmr_score": 0.8542,
      "relevance_score": 0.9123
    },
    ...
  ],
  "cart_summary": {
    "items": ["MAIN_001", "SIDE_001"],
    "item_count": 2,
    "total_value": 380,
    "categories": ["Main", "Side"],
    "diversity_score": 1.0,
    "missing_categories": ["Beverage", "Dessert"]
  },
  "context": {
    "mealtime": "lunch",
    "geohash": "tdr1y",
    "personalized": true
  },
  "latency": {
    "feature_fetch_ms": 2.5,
    "encoding_ms": 45.3,
    "retrieval_ms": 8.2,
    "ranking_ms": 3.1,
    "total_ms": 59.1
  },
  "metadata": {
    "candidates_retrieved": 20,
    "candidates_after_filter": 18,
    "mmr_lambda": 0.7,
    "model_version": "swaadstack-v1.0"
  }
}
```

### Available Item IDs

| Category | IDs                  |
| -------- | -------------------- |
| Main     | MAIN_001 to MAIN_018 |
| Side     | SIDE_001 to SIDE_012 |
| Beverage | BEV_001 to BEV_012   |
| Dessert  | DES_001 to DES_008   |

---

## 🧠 Model Architecture

### Two-Tower Design

| Component           | Input         | Output   | Purpose                                 |
| ------------------- | ------------- | -------- | --------------------------------------- |
| **Item Tower**      | (B, 384)      | (B, 128) | Project item embeddings to shared space |
| **User Tower**      | (B, S, 384)   | (B, 128) | Encode cart sequence as intent vector   |
| **Context Encoder** | (B, 4) + (B,) | (B, 32)  | Temporal + geographic features          |

### Transformer Details

- **Layers**: 2 (Student model, distilled from 12-layer Teacher)
- **Heads**: 4 (multi-head self-attention)
- **Hidden dim**: 256
- **Feedforward dim**: 512
- **Activation**: GELU
- **Normalization**: Pre-LayerNorm (better training stability)
- **Sequence pool**: [CLS] token approach

### Training Strategy

- **Loss**: BCEWithLogitsLoss (binary relevance prediction)
- **Negatives**: 5 per positive (50% hard, 50% easy)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-5)
- **Schedule**: OneCycleLR with 10% warmup + cosine decay
- **Regularization**: Dropout=0.1, gradient clipping=1.0

---

## 📊 Evaluation & Metrics

### Offline Metrics (Section 5.1.2)

| Metric          | What it Measures     | CSAO Alignment                             |
| --------------- | -------------------- | ------------------------------------------ |
| **NDCG@K**      | Ranking quality      | Penalizes relevant items at low positions  |
| **Precision@K** | Top-K accuracy       | Every pixel of the rail drives conversions |
| **Recall@K**    | Coverage             | Ensures meal completion logic is sound     |
| **AUC**         | Discriminatory power | Distinguishes clicks from non-clicks       |

### Slicing Analysis (Section 5.2.2)

- **Temporal slicing**: Performance by mealtime (breakfast/lunch/dinner/late-night)
- **Cohort slicing**: Budget vs. premium user segments
- **Category slicing**: Per-category recommendation accuracy

---

## 🚢 Deployment

### Local Development

```bash
pip install -r requirements.txt
python data_generator.py
python train.py
python app.py
```

### Docker Production

```bash
docker compose up --build -d
```

### Kubernetes (Production @ Scale)

The system is designed for K8s with:

- **HPA**: Scale pods on CPU > 60% or queue depth
- **Redis Cluster**: Sharded by user_id for linear throughput scaling
- **TF-Serving/Triton**: For production model hosting
- **Envoy/Nginx**: API gateway with TLS termination

### Latency Budget

| Component         | Budget         | Technology              |
| ----------------- | -------------- | ----------------------- |
| Network + Gateway | 20-40ms        | HTTP/2, edge caching    |
| Feature Fetch     | 10-15ms        | Redis Pipeline (MGET)   |
| ANN Search        | 15-25ms        | FAISS/ScaNN in-memory   |
| Model Inference   | 50-80ms        | INT8 quantized, ONNX    |
| Business Logic    | 10-20ms        | Optimized Python        |
| **Total**         | **~105-180ms** | **Safe margin: ~120ms** |

---

## 💰 Business Impact

### Projected Metrics

| Metric                     | Target        | Mechanism                              |
| -------------------------- | ------------- | -------------------------------------- |
| **AOV Lift**               | +1.5% to 3.0% | Higher acceptance of relevant add-ons  |
| **Add-on Acceptance Rate** | +5% relative  | Context-aware, diverse recommendations |
| **Cart-to-Order (C2O)**    | Improved      | Reduced decision fatigue               |

### Guardrail Metrics

- Cart Abandonment Rate: Must not increase > 0.5%
- System Latency (P99): Must remain < 300ms
- Order Cancellation Rate: Must remain flat

### A/B Testing: Switchback Design

Uses **time-sliced switchback testing** (not user-level) to avoid SUTVA violations in hyper-local delivery networks.

---

## 🧪 Testing

```bash
# Run all tests
pytest test_swaadstack.py -v

# Run specific test class
pytest test_swaadstack.py::TestMMRReranking -v

# Run with coverage
pytest test_swaadstack.py --cov=. --cov-report=html
```

### Test Coverage

| Category           | Tests    | Coverage                            |
| ------------------ | -------- | ----------------------------------- |
| MMR Re-ranking     | 6 tests  | Diversity, exclusion, edge cases    |
| Temporal Encoding  | 3 tests  | Cyclicality, range, labels          |
| Embedding Utils    | 5 tests  | Normalization, geohash, similarity  |
| Model Architecture | 7 tests  | Shapes, normalization, forward pass |
| Data Processing    | 2 tests  | Padding, masks                      |
| API Integration    | 12 tests | Golden path, cold-start, errors     |
| Performance        | 3 tests  | Latency benchmarks                  |

---

## 📜 License

This project is not for sale and redistribution purposes.

---

_Built with ❤️ for the love of food and AI. By Ramharsh Sanjay Dandekar_
