# Automated Customer Reviews

*Ironhack AI Engineering Bootcamp*

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Goals](#project-goals)
- [Dataset](#datasets)
- [Deliverables](#deliverables)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Results Summary](#results-summary)
- [Technologies Used](#technologies-used)
- [Team](#team)

---

## Team

**Marcos Sousa** - AI Engineering Bootcamp Student  
Ironhack Munich  
February 2025

---

## Project Overview

This project contains an NLP-powered product review aggregator that uses machine learning to analyze Amazon customer reviews and generate actionable insights.

**Timeline**: 4 days

---

## Datasets

All original datasets are publicly available on Kaggle.

You can find them [here](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products).

---

## Project Goals

### Task 1: Sentiment Analysis
Classify customer reviews into three categories (positive, neutral, negative) to help businesses understand customer satisfaction and identify areas for improvement.

**Approach**: Compare baseline (Logistic Regression) vs pre-trained (RoBERTa zero-shot) vs fine-tuned (RoBERTa) models.

### Task 2: Product Clustering
Group Amazon products into 4-6 meaningful meta-categories to simplify navigation and analysis.

**Approach**: Semantic embeddings (Sentence Transformers) + K-Means clustering, evaluated against ground truth using Adjusted Rand Index.

### Task 3: Review Summarization
Generate blog-style recommendation articles for each product category to help customers make informed purchase decisions.

**Approach**: LLaMA 3.2-3B-Instruct with iterative prompt engineering, evaluated using ROUGE, BERTScore, and custom grounding metrics.

---

## Deliverables

### 1. Source Code
- **Notebooks**: 4 Jupyter notebooks covering EDA, sentiment analysis, clustering, and text generation
- **Documentation**: Comprehensive README and inline documentation
- **Repository**: Clean, organized GitHub repository with version control

### 2. Generated Outputs
- **Blog Articles**: `generated_blogposts_v3_no_early_clip.txt` - 10 AI-generated product review articles (5 categories × 2 temperature settings)

### 3. Presentation
- **Slides**: PowerPoint presentation covering methodology, results, and insights

### 4. Demo
- **Static HTML**: Single HTML file with demo hosted publicly in Netlify. Visit the online version [here](https://automated-customer-reviews.netlify.app/).

---

## Repository Structure

```
root/
│
├── data/
│   ├── raw/                                    # Original Amazon review datasets
│   │   ├── 1429_1.csv                          # 34,660 reviews
│   │   ├── amazon_(...)_2017_2018.csv          # 5,000 reviews
│   │   └── amazon_(...)_Feb_April_2019.csv     # 28,332 reviews
│   │
│   └── processed/                   # Cleaned and preprocessed data
│       ├── sentiment_ready.csv      # 39,794 reviews (deduplicated)
│       └── electronics_ready.csv    # 30,429 electronics reviews
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Data exploration and preprocessing
│   ├── 02_sentiment.ipynb           # Sentiment classification (3 models)
│   ├── 03_clustering.ipynb          # Product clustering with K-Means
│   └── 04_generation.ipynb          # LLM-based review summarization
│
├── outputs/
│   ├── models/
│   │   └── generated_blogposts_v3_no_early_clip.txt    # Final blog articles
│   │   
│   │
│   └── figures/                     # Visualization outputs
│       ├── confusion_matrices
│       └── clustering_umap.png
│
│
├── deliverables/
│   └── G8_Presentation.pptx      # Final presentation slides
|   └──
│
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

---

## Quick Start

### Prerequisites
- Python 3.9+
- Google Colab (for GPU access) or local CUDA-enabled GPU
- Hugging Face account (for gated model access)

### Installation

```bash
# Clone the repository
git clone https://github.com/marcosfsousa/project-ironhack-automated-customer-reviews.git

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

**Option 1: Google Colab (Recommended)**
1. Upload notebooks to Google Colab
2. Select L4 GPU runtime: Runtime → Change runtime type → L4 GPU
3. Run cells sequentially

**Option 2: Local Jupyter**
```bash
jupyter notebook
# Open notebooks/01_eda.ipynb and run sequentially
```

**Execution Order**:
1. `01_eda.ipynb` - Data preprocessing (5-10 min)
2. `02_sentiment.ipynb` - Sentiment models (15-20 min with GPU)
3. `03_clustering.ipynb` - Product clustering (5-10 min)
4. `04_generation.ipynb` - Text generation (25-30 min with GPU)

---

## Results Summary

### Sentiment Analysis (Task 1)

| Model | Accuracy | F1 Macro | Key Finding |
|-------|----------|----------|-------------|
| Logistic Regression (Baseline) | 0.697 | 0.685 | Solid baseline with TF-IDF |
| RoBERTa Zero-Shot | 0.636 | 0.590 | Domain mismatch (Twitter → Amazon) |
| **RoBERTa Fine-Tuned** | **0.735** | **0.724** | Best performance, domain-adapted |

**Key Insight**: Fine-tuning improved neutral class F1 from 0.291 → 0.565, demonstrating the importance of domain adaptation.

---

### Product Clustering (Task 2)

- **Method**: Sentence Transformers (`all-MiniLM-L6-v2`) + K-Means (k=5)
- **Categories**: Fire Tablets, Kindle E-Readers, Fire Kids Edition, Echo & Smart Speakers, Fire TV & Streaming
- **Evaluation**: Adjusted Rand Index = 0.42 (moderate alignment with ground truth)
- **Visualization**: UMAP projection shows clear semantic groupings

**Key Insight**: Semantic embeddings successfully captured product similarities beyond simple keyword matching.

---

### Review Summarization (Task 3)

**Final Results (V3)**:
- **Model**: LLaMA 3.2-3B-Instruct
- **BERTScore**: 0.811 (strong semantic alignment)
- **Grounding**: 0.599 (60% vocabulary from source reviews)
- **Compression**: 0.291 (effective condensation)
- **Average Length**: 392 words (target: 350-500)

**Iterative Improvements**:
- **V1 → V2**: Fixed prompt template bug, eliminated cross-category confusion
- **V2 → V3**: Enhanced verdict language, removed noisy categories, improved grounding by 12.7%

**Key Insight**: Prompt engineering and data quality had greater impact than model size. Structured prompts with explicit verdict requirements increased actionable guidance from 7 → 11 mentions.

---

## Technologies Used

### Core Libraries
- **Data Processing**: pandas, numpy, scikit-learn
- **NLP Models**: transformers (Hugging Face), sentence-transformers
- **Deep Learning**: PyTorch, accelerate
- **Evaluation**: rouge-score, bert-score
- **Visualization**: matplotlib, UMAP

### Models
- **Sentiment**: `cardiffnlp/twitter-roberta-base-sentiment-latest` (fine-tuned)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Generation**: `meta-llama/Llama-3.2-3B-Instruct`

### Infrastructure
- **Development**: Google Colab with L4 GPU
- **Version Control**: Git/GitHub
- **Environment**: Python 3.10, CUDA 12.1

---

## Key Learnings

1. **Data Quality > Model Size**: Removing noisy categories improved results more than switching to larger models
2. **Prompt Engineering Matters**: Iterative refinement (V1→V2→V3) had massive impact on generation quality
3. **Domain Adaptation Required**: Zero-shot models underperformed without fine-tuning on target domain
4. **Evaluation Strategy**: Relative metrics (V1 vs V2 vs V3) more informative than absolute scores with imperfect baselines
5. **Class Imbalance**: Weighted loss + balanced test set outperformed undersampling approaches

---

## License

This project is for educational purposes as part of the Ironhack AI Engineering Bootcamp.

---

## Acknowledgments

- Ironhack instructors and teaching assistants
- Hugging Face for model access
- Amazon Customer Reviews dataset (Kaggle)
- Open-source NLP community

---

**Last Updated**: February 20, 2025  