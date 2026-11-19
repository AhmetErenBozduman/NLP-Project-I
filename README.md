# NLP Project I — Structured Text Classification & Information Extraction

This project evaluates structured JSON generation by a locally run large language model. It uses **Llama 3.2 (3B)** through **Ollama** to classify BBC news articles and extract named entities.

## Dataset

- **Source:** [SetFit / BBC News Dataset (Hugging Face)](https://huggingface.co/datasets/SetFit/bbc-news)
- **Evaluation set:** 100 news texts, stored in `dataset.csv`
- **CSV columns:** `id`, `text`, `expected_category`
- **Target categories:** `sports`, `economy`, `technology`, `politics`, `arts`, `other`

## Output format

The program requests a JSON object containing:

- `category`: one of the six target categories
- `entities`: a list of up to five named entities

It checks first-attempt JSON parsing, attempts to clean malformed responses, evaluates category predictions against `expected_category`, checks the required JSON structure, and calculates classification accuracy and macro F1.

## Requirements

- Python 3
- [Ollama](https://ollama.com/) installed and running locally
- Ollama model: `llama3.2:3b`
- Python packages: `ollama`, `scikit-learn`

Install the Python dependencies:

```bash
python -m pip install ollama scikit-learn
```

Download the model:

```bash
ollama pull llama3.2:3b
```

## Run

Keep `main.py` and `dataset.csv` in the same folder, then run:

```bash
python main.py
```

The program prints each article, its generated JSON, validation messages, and aggregate evaluation metrics.

## Experiment settings and results

All experiments used the same 100-text evaluation set, the `llama3.2:3b` model, and seed `42`.

| Metric | Prompt only | JSON Schema enabled | High temperature |
|---|---:|---:|---:|
| Temperature | 0.2 | 0.2 | 1.5 |
| JSON Schema enforcement | Off | On | Off |
| First-attempt valid JSON | 74/100 | 100/100 | 38/100 |
| JSON recovered by cleaning | 0 | 0 | 10 |
| Total parsed JSON | 74/100 | 100/100 | 48/100 |
| Schema-valid outputs | 69/100 | 100/100 | 41/100 |
| Correct / evaluated category predictions | 68/74 | 94/100 | 41/47 |
| Classification accuracy | 91.89% | 94.00% | 87.23% |
| Classification macro F1 | 0.7728 | 0.7908 | 0.7548 |

**Evaluation note:** In the high-temperature experiment, one parsed JSON response was a list rather than an object, so 48 responses were parsed but only 47 were included in the classification evaluation. Classification metrics are calculated on evaluable predictions, not all 100 inputs.

To reproduce the comparisons, edit the `options` temperature value and enable or disable `format=JSON_SCHEMA` in `main.py`. When enabling schema enforcement, ensure that the `JSON_SCHEMA` definition is uncommented as well. Keep the dataset and seed unchanged for comparison.

## Project files

- `main.py` — inference and evaluation script
- `dataset.csv` — 100 evaluation texts and category labels
- `NLP_Project_I_Technical_Report_Final_Revised.pdf` — technical report
- `README.md` — project instructions and experiment summary

## References

- [BBC News Dataset — SetFit, Hugging Face](https://huggingface.co/datasets/SetFit/bbc-news)
- [Meta Llama 3.2](https://www.llama.com/)

## AI usage statement

Generative AI tools were used to support code development.
