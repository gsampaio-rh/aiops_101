# AI for Kubernetes Operations: A Practical Notebook Series

Welcome to the **AI for Kubernetes Operations** repository! This project is a series of interactive Jupyter notebooks designed for professionals working with Kubernetes, OpenShift, Linux, and containerized environments. These notebooks introduce AI/ML concepts step-by-step, applying them to real-world scenarios to optimize Kubernetes workflows.

## Project Overview

This repository aims to bridge the gap between AI/ML and Kubernetes operations. By following this series, you'll learn how to use predictive and generative AI to enhance operational efficiency, improve uptime, and automate routine tasks. 

The notebooks are designed with incremental storytelling, guiding you from foundational AI/ML concepts to advanced topics like anomaly detection, incident prediction, and automated remediation.

## Repository Structure

```bash
.
├── 1-Exploratory-Data-Analysis.ipynb   # Intro to data visualization and pattern discovery in Kubernetes
├── 2-Machine-Learning.ipynb            # Basics of machine learning with structured Kubernetes data
├── 3-Supervised-Learning.ipynb         # Predicting Kubernetes incidents using supervised learning
├── 4-Unsupervised-Learning.ipynb       # Detecting anomalies with clustering techniques
├── 5-Advanced-Anomaly-Detection.ipynb  # Advanced anomaly detection models for Kubernetes
├── 6-Simple-Deep-Learning.ipynb        # Introduction to deep learning with time-series data
├── 7-Advanced-Deep-Learning.ipynb      # Enhancing models with feature engineering and hyperparameter tuning
├── 8-Transformers.ipynb                # Using transformers for time-series predictions
├── 9-LLM.ipynb                         # Leveraging large language models for Kubernetes operational tasks
├── 10-GraphRag.ipynb                   # Graph-based RAG for uncovering ITOps relationships
├── 11-Agents.ipynb                     # Building AI agents for autonomous Kubernetes management
├── README.md                           # This file
├── dataset/                            # Additional dataset files
│   └── kubernetes_operational_data.csv
├── graph/                              # Graph-related files
│   └── post-mortem.txt
├── images/                             # Images used in the notebooks
└── markdown_files/                     # Supporting markdown files
    └── README.md
```

## Getting Started

### Prerequisites

- **Python**: Version 3.8 or higher
- **JupyterLab**: For running the notebooks
- **Dependencies**: Install required Python packages:

  ```bash
  pip install -r requirements.txt
  ```

### Running the Notebooks

1. Clone this repository:

   ```bash
   git clone https://github.com/your-repo/ai-for-k8s.git
   ```

2. Navigate to the project directory:

   ```bash
   cd ai-for-k8s
   ```

3. Start JupyterLab:

   ```bash
   jupyter lab
   ```

4. Open the notebook of your choice to begin.

## Series Outline

### 1. Exploratory Data Analysis (EDA)

Learn how to explore and understand Kubernetes metrics, uncovering data patterns critical for operational decisions.

### 2. Machine Learning Basics

Understand ML fundamentals and how to apply them to Kubernetes data for predictive insights.

### 3. Supervised Learning

Build and evaluate models to predict incidents, empowering proactive Kubernetes management.

### 4. Unsupervised Learning

Discover anomalies in Kubernetes systems using clustering techniques.

### 5. Advanced Anomaly Detection

Dive into cutting-edge anomaly detection algorithms like Isolation Forests and Autoencoders.

### 6. Simple Deep Learning

Get hands-on with LSTM models for time-series predictions in Kubernetes environments.

### 7. Advanced Deep Learning

Enhance model accuracy with feature engineering and hyperparameter optimization.

### 8. Transformers

Leverage transformer architectures for sequential data analysis and forecasting.

### 9. Large Language Models (LLMs)

Use LLMs for log analysis, incident reporting, and troubleshooting automation.

### 10. GraphRAG

Explore Graph-based Retrieval-Augmented Generation to uncover relationships between Kubernetes services and incidents.

### 11. Agents for Autonomous Kubernetes Management

Build AI agents that dynamically monitor, reason, and act on Kubernetes environments. Learn how agents use tools like monitoring APIs, predictive models, and automated workflows to enable self-healing and autonomous scaling in real time.

## Data Sources

The repository includes **synthetic datasets** modeled after real-world Kubernetes operations to ensure relevance while maintaining data privacy. You can find these in the `data/` and `dataset/` directories.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the content, add new notebooks, or suggest features.
