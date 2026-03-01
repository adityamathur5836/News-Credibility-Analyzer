# Project: Intelligent News Credibility Analysis

## From Pattern-Based Classification to Agentic AI Verification

### Project Overview
This project focuses on the development of a multi-phase system for evaluating the credibility of news content. It transitions from a statistical, classical machine learning approach in Milestone 1 to an autonomous, reasoning-based AI agent in Milestone 2.

- **Milestone 1:** Implementation of a supervised machine learning pipeline using classical NLP (TF-IDF) and scikit-learn classifiers (Logistic Regression & Decision Trees) to detect misinformation patterns.
- **Milestone 2:** Extension into an agentic AI system that performs fact-checking, source retrieval (RAG), and structured reasoning to verify claims against external knowledge sources.

---

### Constraints & Requirements
- **Team Size:** 3 Students (Aditya Mathur, Om Kar Shukla, Yachna Khanna)
- **API Budget:** Free Tier Only (Scikit-learn, NLTK, Streamlit)
- **Framework:** Classical ML (M1) / Agentic AI (M2)
- **Hosting:** Streamlit Cloud

---

### Technology Stack
| Component | Technology |
| :--- | :--- |
| **ML Models (M1)** | Logistic Regression, Decision Trees, Scikit-Learn |
| **NLP Pipeline** | NLTK (Lemmatization, Stopword Removal), TF-IDF |
| **UI Framework** | Streamlit |
| **Execution** | Python 3.x |

---

### Milestones & Deliverables

#### Milestone 1: Classical ML Credibility Classification (Current)
**Objective:** Build a robust baseline system that identifies "Fake" vs "True" news using lexical patterns and statistical weighting *without deep learning or LLMs*.

**Key Deliverables:**
- **Pre-processed Dataset:** Cleaned and deduplicated corpus of ~38,644 news articles.
- **System Architecture:** Sequential pipeline: Preprocessing → TF-IDF Vectorization → Inference.
- **Working Application:** Streamlit web UI for real-time credibility assessment.
- **Evaluation Report:** Detailed performance metrics (Accuracy: 98.94%, F1-Score: 99.03%).

#### Milestone 2: Agentic AI Fact-Check Assistant (Upcoming)
**Objective:** Transform the system into an autonomous agent that reasons about specific claims and retrieves evidence from the web to validate news integrity.

**Key Deliverables:**
- **Publicly Deployed App:** Hosted on Streamlit Cloud.
- **Agent Workflow:** Implementation of reasoning loops (Plan → Retrieve → Verify).
- **RAG Integration:** Retrieval-Augmented Generation for grounded factual reporting.
- **Demo Video:** Walkthrough of the agentic verification process.

---

### Evaluation Criteria

| Phase | Weight | Criteria |
| :--- | :--- | :--- |
| **Milestone 1** | 25% | Data Cleaning, TF-IDF Feature Engineering, Model Accuracy/F1, UI Functional Usability. |
| **Milestone 2** | 30% | Agentic Reasoning Quality, RAG Implementation, Fact-Retrieval Accuracy, Successful Deployment. |

> [!IMPORTANT]
> This project adheres to a "Classical-First" philosophy, ensuring a strong statistical baseline is established before moving to complex Generative AI agent architectures.

---

### How to Run (Milestone 1)

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train & Save Model:**
   ```bash
   python 13_train_and_save_model.py
   ```

3. **Launch Streamlit App:**
   ```bash
   streamlit run app.py
   ```
