# Autonomous Financial Transaction Categorization System

A production-ready, full-stack AI system for categorizing financial transactions with:
- **96.15% F1-Score** through ensemble machine learning
- **Zero external API dependencies** for inference
- **Professional web UI** with React + FastAPI backend
- **Dynamic taxonomy configuration** via YAML (150+ Indian brands)
- **Complete explainability** with confidence scores and alternatives
- **Batch processing** and feedback loop for continuous improvement

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Testing](#testing)

## Features

### Core Capabilities

- **Multi-Stage Ensemble Classification**
  - Logistic Regression (interpretable baseline)
  - Support Vector Machine (non-linear boundaries)
  - Random Forest (feature interactions)
  - Soft voting for optimal performance

- **Robust Preprocessing**
  - Text normalization and tokenization
  - TF-IDF + character n-grams
  - Hand-crafted feature engineering
  - Handles typos, case variations, special characters

- **Explainability & Transparency**
  - LIME-based feature attribution
  - Per-prediction confidence scores
  - Individual model vote breakdown
  - Top-K alternative predictions

- **Dynamic Configuration**
  - YAML-based category taxonomy
  - Update categories without code changes
  - Configurable confidence thresholds
  - Runtime category management

- **Ethical AI**
  - Per-category bias monitoring
  - Confusion pattern analysis
  - Text length bias detection
  - Confidence calibration auditing

- **Feedback Loop**
  - Low-confidence prediction flagging
  - User correction logging
  - Retraining data collection
  - Pattern identification

- **Professional Web Interface**
  - React-based frontend with modern UI
  - Real-time single transaction predictions
  - Batch CSV file processing
  - Interactive dashboard with metrics
  - Feedback submission interface

- **Enterprise-Ready API**
  - RESTful FastAPI backend
  - Auto-generated Swagger documentation
  - Batch processing capability
  - Health monitoring endpoints
  - Comprehensive error handling

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSACTION INPUT                         │
│            (raw string, optional metadata)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              PREPROCESSING PIPELINE                           │
│  • Normalization, tokenization, n-gram extraction             │
│  • TF-IDF + character n-grams + hand-crafted features        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│           ENSEMBLE CLASSIFICATION                             │
│  Logistic Regression + SVM + Random Forest                    │
│  Soft voting (probability averaging)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│        EXPLAINABILITY & CONFIDENCE                            │
│  • Confidence score + Top-K alternatives                      │
│  • Feature attribution (LIME)                                 │
│  • Individual model votes                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│           FEEDBACK & UNCERTAINTY HANDLER                      │
│  • Flag low-confidence predictions                            │
│  • Log corrections for retraining                             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                 FINAL PREDICTION OUTPUT                        │
│  {category, confidence, explanation, alternatives}            │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+ (for frontend)
- 4GB+ RAM recommended

### Start the Full Application

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Start Backend (Terminal 1)
cd backend
python -m uvicorn app.main:app --reload
# Backend running at http://localhost:8000

# 3. Start Frontend (Terminal 2)
cd frontend
npm run dev
# Frontend running at http://localhost:3000

# 4. Test the System
curl -X POST "http://localhost:8000/api/v1/predict/single" \
  -H "Content-Type: application/json" \
  -d '{"transaction": "BigBasket grocery delivery"}'
```

### Access the Application
- **Frontend UI**: http://localhost:3000
- **API Documentation**: http://localhost:8000/api/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/api/redoc

## Installation

### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Navigate to backend directory
cd backend
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install
```

## Running the Application

### Frontend Pages

The React frontend provides the following pages:

1. **Dashboard** (`/`) - System metrics and statistics
   - Model performance (F1-score, accuracy)
   - Category distribution
   - System status

2. **Predict** (`/predict`) - Single transaction categorization
   - Enter transaction description
   - Real-time predictions with confidence scores
   - Alternative predictions
   - Processing time display

3. **Batch Upload** (`/batch`) - CSV file processing
   - Upload CSV files with transactions
   - Batch processing results
   - Download categorized results
   - Summary statistics

4. **Feedback** (`/feedback`) - Feedback loop system
   - Report incorrect categorizations
   - Help improve the model
   - Track feedback submissions

### API Endpoints

#### Health Check
```bash
GET http://localhost:8000/api/v1/health
```

#### Single Transaction Prediction
```bash
POST http://localhost:8000/api/v1/predict/single
Content-Type: application/json

{
  "transaction": "BigBasket grocery delivery"
}
```

#### Batch Processing
```bash
POST http://localhost:8000/api/v1/batch/upload
Content-Type: multipart/form-data

file: transactions.csv
```

#### Feedback Submission
```bash
POST http://localhost:8000/api/v1/feedback
Content-Type: application/json

{
  "transaction": "Nykaa beauty products",
  "predicted_category": "Food & Dining",
  "correct_category": "Shopping",
  "confidence": 0.183
}
```

#### Get Taxonomy
```bash
GET http://localhost:8000/api/v1/taxonomy
```

#### Get Metrics
```bash
GET http://localhost:8000/api/v1/metrics
```

## Performance Metrics

The system achieves production-level accuracy with:

- **Macro F1-Score**: 96.15% (exceeds 90% requirement)
- **Accuracy**: 96.13%
- **Coverage**: 150+ Indian brands
- **Inference Latency**: ~100ms per prediction
- **Batch Throughput**: ~1,000 transactions/second

### Reproducibility

- **Fixed Random Seeds**: All random operations seeded for consistency
- **Deterministic Data Split**: Stratified sampling ensures balanced datasets
- **Model Persistence**: Trained models saved with joblib for exact reproduction
- **Configuration Management**: YAML-based taxonomy for version control

## Project Structure

```
ghci/
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── main.py                   # FastAPI application
│   │   ├── api/v1/
│   │   │   ├── predict.py            # Prediction endpoints
│   │   │   ├── batch.py              # Batch processing
│   │   │   ├── feedback.py           # Feedback endpoints
│   │   │   ├── taxonomy.py           # Taxonomy endpoints
│   │   │   ├── metrics.py            # Metrics endpoints
│   │   │   └── health.py             # Health check
│   │   ├── core/
│   │   │   └── model_loader.py       # ML model integration
│   │   ├── services/
│   │   │   └── insights_engine.py    # Insights generation
│   │   └── models/                   # Pydantic schemas
│   └── requirements.txt              # Python dependencies
│
├── frontend/                         # React + Vite Frontend
│   ├── src/
│   │   ├── App.jsx                   # Main app component
│   │   ├── components/
│   │   │   ├── Header.jsx            # Header component
│   │   │   ├── Sidebar.jsx           # Navigation sidebar
│   │   │   └── ConfidenceMeter.jsx   # Confidence visualization
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx         # System dashboard
│   │   │   ├── Predict.jsx           # Single prediction
│   │   │   ├── BatchUpload.jsx       # Batch processing
│   │   │   ├── Feedback.jsx          # Feedback submission
│   │   │   └── ModelPerformance.jsx  # Model metrics
│   │   ├── services/
│   │   │   └── api.js                # API client
│   │   └── index.css                 # Global styles
│   ├── package.json                  # Node dependencies
│   └── vite.config.js                # Vite configuration
│
├── src/                              # Core ML Modules
│   ├── preprocessing.py              # Text preprocessing
│   ├── models.py                     # Model classes
│   ├── explainer.py                  # LIME explainability
│   ├── taxonomy_loader.py            # Configuration loader
│   ├── feedback_handler.py           # Feedback processing
│   └── bias_auditor.py               # Bias monitoring
│
├── config/
│   └── taxonomy.yaml                 # Category configuration (150+ brands)
│
├── models/                           # Trained ML models
│   ├── voting_ensemble.pkl           # Ensemble classifier
│   ├── preprocessor.pkl              # TF-IDF vectorizer
│   └── metadata.pkl                  # Model metadata
│
├── data/
│   ├── raw/                          # Raw data
│   ├── processed/                    # Processed datasets
│   └── source_documentation.md       # Data documentation
│
├── scripts/                          # Utility scripts
│   ├── generate_data.py              # Data generation
│   ├── train.py                      # Model training
│   ├── evaluate.py                   # Model evaluation
│   └── predict.py                    # CLI prediction
│
├── tests/                            # Unit tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_taxonomy_loader.py
│
├── reports/                          # Evaluation reports
│   ├── evaluation_report.md
│   └── confusion_matrix.png
│
├── requirements.txt                  # Python dependencies
├── Solution.md                       # Architecture & design
└── README.md                         # This file
```

## Configuration

### Taxonomy Configuration (config/taxonomy.yaml)

```yaml
version: "1.0"
categories:
  Food & Dining:
    keywords:
      - starbucks
      - mcdonalds
      - restaurant
    aliases:
      - dining
      - food
    confidence_boost: 0.05

confidence_thresholds:
  high: 0.85
  medium: 0.70
  low: 0.60

bias_mitigation:
  monitor_amount: true
  monitor_region: true
  enforce_balanced_accuracy: true
```

### Adding New Categories

Edit `config/taxonomy.yaml`:

```yaml
categories:
  New Category:
    keywords:
      - keyword1
      - keyword2
    confidence_boost: 0.03
```

No code changes or retraining required!

## API Reference

### Response Schema Examples

#### Prediction Response
```json
{
  "transaction": "BigBasket grocery delivery",
  "predicted_category": "Groceries",
  "confidence": 0.563,
  "alternatives": [
    {
      "category": "Fuel",
      "confidence": 0.275
    },
    {
      "category": "Health",
      "confidence": 0.038
    }
  ],
  "metadata": {
    "processing_time_ms": 103.97,
    "model_version": "1.0.0",
    "timestamp": "2025-11-21T10:30:00Z"
  }
}
```

#### Batch Upload Response
```json
{
  "job_id": "batch_abc123",
  "total_transactions": 100,
  "status": "completed",
  "summary": {
    "processed": 100,
    "high_confidence": 75,
    "medium_confidence": 20,
    "low_confidence": 5
  },
  "download_url": "/api/v1/batch/batch_abc123/download"
}
```

### Python SDK (Optional)

For programmatic integration in Python:

```python
from src.preprocessing import TransactionPreprocessor
from src.models import TransactionClassifier
from src.explainer import TransactionExplainer

# Load models and preprocessor
preprocessor = TransactionPreprocessor(max_features=5000, seed=42)
classifier = TransactionClassifier(seed=42)
explainer = TransactionExplainer(classifier, preprocessor)

# Make predictions
result = classifier.predict_with_confidence("BigBasket delivery", top_k=3)
explanation = explainer.explain_prediction("BigBasket delivery")
```

## Testing

Run unit tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Key Design Decisions

1. **Ensemble over Single Model**: Combines strengths of multiple algorithms
2. **TF-IDF + Character N-grams**: Handles typos and variations robustly
3. **No External APIs**: Fully self-contained, no network dependencies
4. **YAML Configuration**: User-friendly, version-controllable taxonomy
5. **LIME for Explainability**: Model-agnostic, locally accurate explanations
6. **Stratified Splits**: Ensures balanced category representation

## Technical Stack

### Backend
- **Framework**: FastAPI (async, auto-docs)
- **ML Models**: Scikit-learn ensemble (Logistic Regression + SVM + Random Forest)
- **Features**: TF-IDF + character n-grams
- **Documentation**: Swagger UI + ReDoc

### Frontend
- **Framework**: React 18 + Vite
- **Routing**: React Router v6
- **HTTP Client**: Axios
- **Styling**: Custom CSS with variables

### ML Pipeline
- **Preprocessing**: Text normalization, TF-IDF vectorization
- **Models**: Soft voting ensemble for optimal performance
- **Explainability**: Confidence scores and alternative predictions
- **Monitoring**: Per-category performance tracking

## Future Enhancements

- Hierarchical category taxonomy
- Advanced NLP models (BERT) for future versions
- Mobile app (iOS/Android)
- WebSocket support for real-time updates
