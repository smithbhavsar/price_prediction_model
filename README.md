# Price Prediction Model

**Price Prediction Model** is a machine learning project designed to estimate product prices based on their descriptions. It leverages fine-tuned large language models (LLMs) such as Llama 3.1 and GPT-4o-mini, an agent-based framework for specialized price prediction tasks, and advanced data visualizations to analyze pricing trends. The project includes data curation, model testing, fine-tuning, and a scalable pricer service, making it suitable for applications in e-commerce and market analysis.

Developed over several months in early 2025, this repository showcases a complete pipeline from data preparation to model deployment, with a focus on efficient fine-tuning using LoRA (Low-Rank Adaptation) and robust evaluation using custom testing frameworks.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Curation](#data-curation)
  - [Model Testing](#model-testing)
  - [Fine-Tuning](#fine-tuning)
  - [Agent Framework](#agent-framework)
  - [Pricer Service](#pricer-service)
- [Visualizations](#visualizations)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Project Overview

The Price Prediction Model project aims to accurately predict product prices from textual descriptions, leveraging both frontier LLMs and traditional machine learning techniques. Key components include:

- **Data Curation**: Processes raw product data into structured datasets (`train.pkl`, `test.pkl`) for training and evaluation.
- **Model Testing**: Evaluates multiple LLMs (e.g., Llama 3.1, GPT-4o-mini, Claude, Qwen) using a custom `Tester` class to compute metrics like average error and RMSLE.
- **Fine-Tuning**: Fine-tunes Llama 3.1 using LoRA and GPT-4o-mini via OpenAI’s API to improve price prediction accuracy.
- **Agent Framework**: Implements a modular agent system (e.g., `specialist_agent.py`, `ensemble_agent.py`) for task-specific price estimation and deal analysis.
- **Pricer Service**: Provides a scalable service (`pricer_service.py`) for real-time price predictions using a product database.

The project was developed iteratively from January to April 2025, with a focus on efficient training on consumer hardware and robust evaluation.

## Repository Structure

The repository is organized into four main directories, each serving a distinct purpose:

```
price_prediction_model/
├── data_curation/                # Data preparation and visualization scripts
│   ├── items.py                  # Defines the Item class for product data
│   ├── loaders.py                # Data loading utilities
│   ├── main.py                   # Main script for data curation
│   ├── categories.png            # Visualization of product categories
│   ├── categories_new.png        # Updated category visualization
│   ├── price_vs_characters.png   # Price vs. description length plot
│   ├── prices.png                # Initial price distribution visualization
│   ├── prices_new.png            # Updated price distribution
│   ├── prices_test.png           # Test set price visualization
│   ├── tokens.png                # Token count analysis
├── frontier_model_test/          # Model testing and benchmarking
│   ├── human_input.csv           # Human-provided input samples
│   ├── human_output.csv          # Human-provided output samples
│   ├── human_pricer.png          # Visualization of human predictions
│   ├── items.py                  # Item class for testing
│   ├── main.py                   # Main script for running tests
│   ├── testing.py                # Tester class for model evaluation
│   ├── *.png                     # Model performance visualizations (e.g., gpt-4o-mini.png, llama-3.3-70b-versatile.png)
├── frontier_model_tuning/        # Fine-tuning scripts and data
│   ├── fine_tune_train.jsonl     # Training data for fine-tuning
│   ├── fine_tune_validation.jsonl # Validation data for fine-tuning
│   ├── gpt_fine_tuned.png        # Visualization of fine-tuned GPT-4o-mini performance
│   ├── items.py                  # Item class for fine-tuning
│   ├── main.py                   # Main fine-tuning script
│   ├── testing.py                # Testing utilities for fine-tuned models
├── price_prediction/             # Agent framework and pricer service
│   ├── agents/                   # Agent-based prediction modules
│   │   ├── __init__.py
│   │   ├── agent.py              # Base agent class
│   │   ├── deals.py              # Deal analysis utilities
│   │   ├── ensemble_agent.py     # Ensemble prediction agent
│   │   ├── frontier_agent.py     # Agent for frontier LLMs
│   │   ├── messaging_agent.py    # Agent for communication tasks
│   │   ├── planning_agent.py     # Agent for planning predictions
│   │   ├── random_forest_agent.py # Random forest-based agent
│   │   ├── scanner_agent.py      # Agent for scanning product data
│   │   ├── specialist_agent.py   # Specialized price prediction agent
│   ├── base.py                   # Base utilities for price prediction
│   ├── deal_agent_framework.py   # Framework for deal-focused agents
│   ├── ensemble-model.png        # Visualization of ensemble model performance
│   ├── gpt-4o-mini-with-context.png # GPT-4o-mini performance with context
│   ├── items.py                  # Item class for price prediction
│   ├── log_utils.py              # Logging utilities
│   ├── logger.py                 # Logger configuration
│   ├── main.py                   # Main script for price prediction
│   ├── memory.json               # Memory storage for agents
│   ├── memory.py                 # Memory management module
│   ├── pricer_service.py         # Scalable price prediction service
│   ├── product_database.py       # Product database management
│   ├── random-forest-model.png   # Random forest model performance
│   ├── testing.py                # Testing utilities for price prediction
│   ├── visualize_chromadb_2d.html # 2D visualization of ChromaDB embeddings
│   ├── visualize_chromadb_2d.py  # Script for 2D ChromaDB visualization
│   ├── visualize_chromadb_3d.html # 3D visualization of ChromaDB embeddings
│   ├── visualize_chromadb_3d.py  # Script for 3D ChromaDB visualization
├── get_folder.py                 # Utility script for folder operations
├── rebuild_git_history.sh        # Script to rebuild Git history with backdated commits
```

## Key Features

- **Data Curation**: Generates structured datasets (`train.pkl`, `test.pkl`) from raw product data, with visualizations for price distributions, categories, and token counts.
- **Model Evaluation**: Benchmarks multiple LLMs (e.g., Llama 3.1, GPT-4o-mini, Claude, Qwen) using a custom `Tester` class, producing metrics and visualizations.
- **Fine-Tuning**: Supports LoRA-based fine-tuning of Llama 3.1 and API-based fine-tuning of GPT-4o-mini for improved price prediction accuracy.
- **Agent Framework**: Modular agents (`ensemble_agent.py`, `specialist_agent.py`) for task-specific price estimation, including ensemble and random forest approaches.
- **Pricer Service**: A scalable service (`pricer_service.py`) for real-time price predictions, integrated with a product database (`product_database.py`).
- **Visualizations**: Comprehensive plots for data analysis (e.g., `prices.png`) and model performance (e.g., `gpt-4o-mini-with-context.png`), including interactive 2D/3D ChromaDB embeddings.

## Installation

### Prerequisites
- **Python**: 3.11.12 or higher
- **Hardware**: GPU with at least 12-16 GB VRAM recommended for Llama 3.1 fine-tuning (CPU fallback is slow).
- **Ollama**: For local Llama 3.1 inference, install `ollama.exe` and pull `llama3.1:8b` (see [Ollama](https://ollama.com/)).
- **Hugging Face Access**: Required for Llama 3.1 fine-tuning. Request access to `meta-llama/Llama-3.1-8B-Instruct` at [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and set your `HF_TOKEN` environment variable.
- **Git**: For cloning the repository.

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/smithbhavsar/price_prediction_model.git
   cd price_prediction_model
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not provided, install the following:
   ```bash
   pip install torch==2.1.0 transformers==4.36.0 peft==0.7.1 bitsandbytes==0.41.3 accelerate==0.25.0 trl==0.7.10 datasets==2.16.1 requests matplotlib numpy pickle5
   ```

4. **Set Environment Variables**:
   Create a `.env` file in the root directory:
   ```plaintext
   HF_TOKEN=your_hugging_face_token
   ```
   Load it using:
   ```bash
   source .env
   ```

5. **Start Ollama (Optional)**:
   For local Llama 3.1 inference:
   ```bash
   ollama run llama3.1:8b
   ```

## Usage

### Data Curation
Generate structured datasets and visualizations:
```bash
python data_curation/main.py
```
- **Inputs**: Raw product data (assumed to be preprocessed into `train.pkl` and `test.pkl`).
- **Outputs**: Visualizations (`prices.png`, `categories.png`, etc.) in `data_curation/`.
- **Key Files**:
  - `items.py`: Defines the `Item` class for product data.
  - `loaders.py`: Utilities for loading and processing data.
  - `main.py`: Orchestrates data curation.

### Model Testing
Evaluate multiple LLMs on the test dataset:
```bash
python frontier_model_test/main.py
```
- **Inputs**: `test.pkl` (test dataset).
- **Outputs**: Performance metrics and visualizations (e.g., `gpt-4o-mini.png`, `llama-3.3-70b-versatile.png`) in `frontier_model_test/`.
- **Key Files**:
  - `testing.py`: Contains the `Tester` class for computing metrics (average error, RMSLE, hits).
  - `items.py`: Item class for test data.
  - `main.py`: Runs the testing pipeline.

### Fine-Tuning
Fine-tune Llama 3.1 using LoRA or GPT-4o-mini via OpenAI’s API:
```bash
python frontier_model_tuning/main.py
```
- **Inputs**: `fine_tune_train.jsonl`, `fine_tune_validation.jsonl`.
- **Outputs**: Fine-tuned model weights (saved to `./results`) and performance visualization (`gpt_fine_tuned.png`).
- **Key Files**:
  - `main.py`: Fine-tuning script (supports LoRA for Llama 3.1).
  - `testing.py`: Evaluates fine-tuned models.
  - **Note**: For Llama 3.1, ensure access to `meta-llama/Llama-3.1-8B-Instruct`. For GPT-4o-mini, an OpenAI API key is required.

### Agent Framework
Run the agent-based price prediction system:
```bash
python price_prediction/main.py
```
- **Inputs**: Product descriptions and `product_database.py`.
- **Outputs**: Predicted prices, stored in `memory.json` or logged via `logger.py`.
- **Key Files**:
  - `agents/`: Modular agents (`ensemble_agent.py`, `specialist_agent.py`, etc.) for task-specific predictions.
  - `deal_agent_framework.py`: Framework for deal analysis.
  - `base.py`, `items.py`: Core utilities and data structures.

### Pricer Service
Deploy the scalable pricer service:
```bash
python price_prediction/pricer_service.py
```
- **Inputs**: Product data via `product_database.py`.
- **Outputs**: Real-time price predictions.
- **Key Files**:
  - `pricer_service.py`: Main service logic.
  - `product_database.py`: Manages product data storage.

## Visualizations

The project includes extensive visualizations to analyze data and model performance:

- **Data Visualizations** (`data_curation/`):
  - `prices.png`, `prices_new.png`, `prices_test.png`: Price distributions.
  - `categories.png`, `categories_new.png`: Product category breakdowns.
  - `price_vs_characters.png`: Price vs. description length.
  - `tokens.png`: Token count analysis.

- **Model Performance** (`frontier_model_test/`):
  - `gpt-4o-mini.png`, `llama-3.3-70b-versatile.png`, etc.: Scatter plots and metrics for each model.
  - `human_pricer.png`: Human baseline performance.

- **Fine-Tuning Results** (`frontier_model_tuning/`):
  - `gpt_fine_tuned.png`: Fine-tuned model performance.

- **Agent Visualizations** (`price_prediction/`):
  - `ensemble-model.png`: Ensemble model performance.
  - `random-forest-model.png`: Random forest agent performance.
  - `gpt-4o-mini-with-context.png`: GPT-4o-mini with context performance.
  - `visualize_chromadb_2d.html`, `visualize_chromadb_3d.html`: Interactive 2D/3D visualizations of ChromaDB embeddings.

To view HTML visualizations:
```bash
open price_prediction/visualize_chromadb_2d.html
open price_prediction/visualize_chromadb_3d.html
```

## License

This project is licensed under the MIT License, except for components using Llama 3.1, which are subject to the [Llama 3.1 Community License](https://llama.meta.com/llama3_1/license). Specifically:

- The fine-tuning scripts in `frontier_model_tuning/` using Llama 3.1 (`meta-llama/Llama-3.1-8B-Instruct`) adhere to the Llama 3.1 Community License, requiring attribution (“Built with Llama”) and compliance with the [Acceptable Use Policy](https://llama.meta.com/llama3_1/use-policy).
- All other components (data curation, agent framework, pricer service) are under the MIT License.

See the `LICENSE` file (if present) for details.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

Please ensure your code follows the project’s style and includes tests where applicable. For issues or feature requests, use the [GitHub Issues](https://github.com/smithbhavsar/price_prediction_model/issues) page.

## Contact

For questions or support, contact the project maintainer:
- **GitHub**: [smithbhavsar](https://github.com/smithbhavsar)

---

*Project developed by Smith Bhavsar, January–April 2025.*