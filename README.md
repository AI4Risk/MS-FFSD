# 🛡️ Behavior-Grounded Semantic Enrichment for Financial Fraud Modeling and Reasoning
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10 or later">
  <img src="https://img.shields.io/badge/Dataset-MS--FFSD-0A7E8C?style=flat-square&logo=databricks&logoColor=white" alt="MS-FFSD Dataset">
  <img src="https://img.shields.io/badge/Generation-Multimodal-7C3AED?style=flat-square" alt="Multimodal Generation">
  <img src="https://img.shields.io/badge/LLM-Enabled-10B981?style=flat-square" alt="LLM Enabled">
</p>

📌 This repository builds the multimodal MS-FFSD dataset from S-FFSD transactions, local IEEE-CIS reference data, statistical initialization tables, consistency priors, and LLM-generated descriptions.

<p align="center">
  <a href="#-structure">📁 Structure</a> ·
  <a href="#️-requirements">⚙️ Requirements</a> ·
  <a href="#-inputs">📥 Inputs</a> ·
  <a href="#️-run">▶️ Run</a> ·
  <a href="#-output">📦 Output</a> ·
  <a href="#-transactioncsv">💳 Transactions</a> ·
  <a href="#-usercsv">👤 Users</a> ·
  <a href="#-merchantcsv">🏪 Merchants</a> ·
  <a href="#-description-tables">💬 Descriptions</a>
</p>

## 📁 Structure

```text
input/
  S-FFSD.csv                         Raw S-FFSD transaction table
  ieee/                              Local-only IEEE-CIS input files
dataset/
  MS-FFSD.zip                        Final five-table release
timestamp_synthesis/                 Temporal Agent: timestamp synthesis
initialization/                      User Agent & Merchant Agent: user, location, and merchant initialization
consistency_optimization/
  code/                              Consistency Agent: alternating consistency optimization
  priors/                            Four priors used by Module 3
long_text_generation/                Textual Agent: profile construction and description generation
work/                                Generated intermediates, checkpoints, and reports
publish_dataset.py                   Final schema conversion, validation, and packaging
run_pipeline.py                      One-command entry point for every module
```

`work/`, the S-FFSD CSV and the IEEE-CIS CSV files are excluded from version control. 

## ⚙️ Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`
- An OpenAI-compatible chat-completions endpoint for Module 4

```bash
pip install -r requirements.txt
```

## 📥 Inputs

Download `S-FFSD.zip` from the `data` directory of the official [AI4Risk/antifraud](https://github.com/AI4Risk/antifraud) repository. Extract the archive and place the resulting file at:

```text
input/S-FFSD.csv
```

The S-FFSD table must contain:


```text
Time, Source, Target, Amount, Location, Type, Labels
```

Download the training data from the official [IEEE-CIS Fraud Detection competition data page](https://www.kaggle.com/competitions/ieee-fraud-detection/data). Place these two downloaded files locally:

```text
input/ieee/train_transaction.csv
input/ieee/train_identity.csv
```

IEEE-CIS provides a reference for realistic transaction-time rhythms, including intraday, weekly, and calendar patterns. Module 1 uses these patterns to convert the relative transaction order in S-FFSD into plausible physical datetimes.

The repository already includes the statistical tables under `initialization/informations/` and the four active consistency priors under `consistency_optimization/priors/`.

## ▶️ Run

Run commands from the repository root. Each module has one entry command:

```bash
python run_pipeline.py timestamp
python run_pipeline.py initialize
python run_pipeline.py optimize
python run_pipeline.py describe --base-url "$LLM_BASE_URL" --api-key "$LLM_API_KEY" --model "$LLM_MODEL"
python run_pipeline.py publish
```

The API options can instead be supplied through `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`. API configuration is intentionally blank in the source code.

Run the complete pipeline with:

```bash
python run_pipeline.py all --base-url "$LLM_BASE_URL" --api-key "$LLM_API_KEY" --model "$LLM_MODEL"
```

Module 4 creates single-transaction user descriptions with deterministic templates and calls the LLM only for multi-transaction users and merchants. Interrupted LLM generation resumes from checkpoints in `work/checkpoints/`.

## 📦 Output

`python run_pipeline.py publish` validates all foreign keys, row counts, required values, transaction ordering, immutable transaction fields, and English-only public text before creating:

```text
dataset/MS-FFSD.zip
```

The archive contains exactly five CSV files:

| File | Columns |
| --- | --- |
| `transaction.csv` | `Datetime`, `Source`, `Target`, `Amount`, `Location`, `Type`, `Labels`, `Transaction_province` |
| `user.csv` | `Source`, `Province`, `Gender`, `Age_group`, `Education`, `Occupation_industry` |
| `merchant.csv` | `Target`, `MCC_level1`, `MCC_level2`, `Merchant_name` |
| `user_description.csv` | `Source`, `User_description` |
| `merchant_description.csv` | `Target`, `Merchant_description` |

### 💳 `transaction.csv`

Each row represents one transaction. `Source` and `Target` are foreign keys to `user.csv` and `merchant.csv`.

| Field | Description |
| --- | --- |
| `Datetime` | Synthesized physical date and time constructed using IEEE-CIS-based temporal priors. |
| `Source` | Anonymized user identifier. |
| `Target` | Anonymized merchant identifier. |
| `Amount` | Original transaction amount from S-FFSD; it is not modified by the generation pipeline. |
| `Location` | Original anonymized transaction-location identifier from S-FFSD. |
| `Type` | Original anonymized transaction-type identifier from S-FFSD. |
| `Labels` | Fraud supervision label: `0` for normal, `1` for fraud, and `2` for unlabeled. |
| `Transaction_province` | Province assigned as the physical location of the transaction. |

### 👤 `user.csv`

Each row contains the structured profile of one user.

| Field | Description |
| --- | --- |
| `Source` | Anonymized user identifier and primary key. |
| `Province` | User's assigned home province. |
| `Gender` | User gender category. |
| `Age_group` | User age-group category. |
| `Education` | User education-level category. |
| `Occupation_industry` | Industry associated with the user's occupation. |

### 🏪 `merchant.csv`

Each row contains the structured profile of one merchant.

| Field | Description |
| --- | --- |
| `Target` | Anonymized merchant identifier and primary key. |
| `MCC_level1` | Broad merchant consumption and behavior category. |
| `MCC_level2` | Fine-grained merchant category within `MCC_level1`. |
| `Merchant_name` | Synthetic English merchant name. |

### 💬 Description tables

| File and field | Description |
| --- | --- |
| `user_description.csv` / `Source` | Foreign key to `user.csv`. |
| `user_description.csv` / `User_description` | Synthetic English narrative summarizing the user's attributes and transaction behavior. |
| `merchant_description.csv` / `Target` | Foreign key to `merchant.csv`. |
| `merchant_description.csv` / `Merchant_description` | Synthetic English narrative describing the merchant's business and customer or transaction profile. |
