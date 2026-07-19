# 🏦 AI_Driven Consumer Complaint Management System 

Automatically sort consumer complaints into the right category — comparing recurrent neural networks against a fine-tuned Transformer to find the most reliable classifier.

## Why This Project Exists

Manually reading and categorizing thousands of consumer complaints is slow and error-prone. This project trains and compares multiple deep learning models that read complaint text and predict its category, so teams can route and prioritize cases automatically instead of by hand.

## Model Showdown

| Model | Type | Handles Long Text | Result |
|---|---|---|---|
| SimpleRNN | Recurrent | Poorly | Weakest performer — used as baseline |
| LSTM | Gated Recurrent | Well | Noticeable jump over baseline |
| GRU | Gated Recurrent (lighter) | Well | Close to LSTM, a bit lighter |
| **Fine-tuned Transformer** | Pretrained + Fine-tuned | Excellently |  **Top performer, picked as final model** |

## How It Works

1. **Data prep** — split complaints into train / validation / test sets
2. **Text cleaning** — normalize, tokenize, and pad/truncate sequences
3. **Training** — each model tuned separately (learning rate, embedding size, sequence length)
4. **Benchmarking** — accuracy, precision, recall, and F1-score compared side by side
5. **Final pick** — best model saved for real-world inference

## What's Inside

```
├── Consumer Complaint Classification.ipynb   # Full training & evaluation workflow
├── app.py
└── README.md
```

## Built With

TensorFlow/Keras · Hugging Face Transformers · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn

## Quick Start

```bash
pip install tensorflow transformers scikit-learn pandas numpy
python app.py
```

Pass any complaint text in, and the model returns the predicted category along with a confidence score.

## Outcome

The fine-tuned Transformer beat every recurrent model by a clear margin in accuracy and F1-score. Its ability to understand context across the full complaint text — not just nearby words — made it the obvious choice for deployment.

## Where This Could Go Next

- Grow the dataset with more categories and harder edge cases
- Add attention visualization so predictions are explainable
- Wrap the winning model in a simple web/API interface for live use
- Export to ONNX for faster, lighter production inference
