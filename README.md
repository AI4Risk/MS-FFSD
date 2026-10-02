# 🛡️ Behavior-Grounded Semantic Enrichment for Financial Fraud Modeling and Reasoning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10 or later">
  <img src="https://img.shields.io/badge/Dataset-MS--FFSD-0A7E8C?style=flat-square&logo=databricks&logoColor=white" alt="MS-FFSD Dataset">
  <img src="https://img.shields.io/badge/Multimodal-Tabular%20%2B%20Text-7C3AED?style=flat-square" alt="Multimodal Dataset">
  <img src="https://img.shields.io/badge/LLM-Enabled-10B981?style=flat-square" alt="LLM Enabled">
</p>

<p>
  <a href="./"><strong>💻 Code</strong></a> :
  We propose a multi-agent semantic enrichment framework that generates interpretable financial semantics grounded in transaction behavior through specialized agents and collaborative refinement.
  <br><br>
  <a href="./dataset/"><strong>🗂️ Dataset</strong></a> :
  We newly contribute a valuable multimodal financial fraud dataset, MS-FFSD, enriched with structured semantics and textual semantics while preserving real-data-grounded transaction behavior.
  <br><br>
  <a href="https://arxiv.org/abs/2609.34211"><strong>📄 Paper</strong></a> :
  Our accompanying paper is available on arXiv and is currently under submission.
</p>

<p align="center">
  This repository provides the complete semantic enrichment framework and the released MS-FFSD dataset.
</p>
<p align="center">
  <a href="#framework-overview">🧩 Framework</a> ·
  <a href="#dataset-overview">🗂️ Dataset</a> ·
  <a href="#label-usage">🏷️ Labels</a> ·
  <a href="#running-framework">🔧 Run</a> ·
  <a href="#acknowledgements">🙏 Acknowledgements</a> ·
  <a href="#contributing">🤝 Contributing</a> · 
  <a href="#citation">📚 Citation</a>
</p>


<a id="framework-overview"></a>

## 🧩 Framework Overview

<p align="center">
  <img src="assets/model.png" width="75%" alt="Multi-agent semantic enrichment framework">
</p>


<p align="center">
  <em>Overview of the multi-agent semantic enrichment framework.</em>
</p>


We use a multi-agent semantic enrichment framework grounded in transaction behavior. 

- The Temporal Agent, User Agent, and Merchant Agent construct temporal, user, and merchant semantics, respectively.
- The Consistency Agent iteratively refines cross-entity semantic consistency over observed user–merchant interactions.
- The Textual Agent then generates entity-level descriptions from the refined structured state.

Throughout the enrichment process, the original transaction structure and fraud labels are preserved.


<a id="dataset-overview"></a>

## 🗂️ Dataset Overview

🚀**Quick Start** : The released MS-FFSD is available at `dataset/MS-FFSD.zip`

<p align="center">
  <img src="assets/table.png" width="75%" alt="MS-FFSD dataset organization">
</p>
<p align="center">
  <em>Organization of the five released CSV files in MS-FFSD.</em>
</p>


💳 *transaction.csv*

- `Datetime` : Synthesized physical date and time constructed using IEEE-CIS-based temporal priors.
- `Source` : User identifier and foreign key to the user table.
- `Target` : Merchant identifier and foreign key to the merchant table.
- `Amount` : Original transaction amount preserved during semantic enrichment.
- `Location` : Original transaction-location identifier.
- `Type` : Original transaction-type identifier.
- `Transaction_province` : Assigned physical province of the transaction.
- `Labels` : Transaction label indicating normal, fraudulent, or unlabeled status.

👤 *user.csv*

- `Source` : User identifier and primary key.
- `Province` : Assigned home province of the user.
- `Gender` : User gender category.
- `Age_group` : User age-group category.
- `Education` : User education-level category.
- `Occupation_industry` : Industry associated with the user's occupation.

🏪 *merchant.csv*

- `Target` : Merchant identifier and primary key.
- `MCC_level1` : Broad merchant consumption and business category.
- `MCC_level2` : Fine-grained merchant category within `MCC_level1`.
- `Merchant_name` : Synthetic English merchant name.

📝 *user_description.csv*

- `Source` : Foreign key to the user table.
- `User_description` : Textual description of the user's consumption behavior and semantic profile.

🏬 *merchant_description.csv*

- `Target` : Foreign key to the merchant table.
- `Merchant_description` : Textual description of the merchant's business characteristics and behavioral context.


<a id="label-usage"></a>

## 🏷️ Label Usage

MS-FFSD contains three transaction labels:

- `1`: confirmed fraud;
- `0`: confirmed normal;
- `2`: unlabeled.

The distinction between confirmed normal and unlabeled transactions reflects the data collection process of the underlying real-world financial data. Fraudulent transactions were labeled after confirmation, while transactions explicitly verified as non-fraudulent were labeled as normal. The remaining transactions were retained as unlabeled.

### 🔄 Using Unlabeled Samples

- **Models or training protocols that can exploit unlabeled samples.**  

  We recommend preserving the unlabeled status. In our graph-based experiments, only confirmed normal and confirmed fraudulent nodes are used as supervised target nodes and are split into training, validation, and test sets. Unlabeled nodes are excluded from the supervised objective, but remain in the graph and participate in message passing, allowing their feature and structural information to contribute to representation learning.

- **Purely supervised models or training protocols that would otherwise leave unlabeled samples unused.**  

  Unlabeled samples can be merged into the normal class and treated as normal/background data. This is consistent with the operational setting of financial fraud detection, where confirmed fraud cases constitute the positive class, while the large volume of transactions without confirmed fraud labels is typically used as non-fraud background data.

> **Note.** Treating unlabeled samples as normal/background data is a modeling convention for making use of unlabeled data, rather than an assertion that every unlabeled transaction has been explicitly verified as normal.


<a id="running-framework"></a>

## 🔧 Running the Framework

The semantic enrichment framework is implemented in this repository using S-FFSD as the transaction data and IEEE-CIS together with the provided statistical priors as external references. The following instructions describe the environment, required inputs, and commands for running the framework.

### ⚙️ Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`
- An OpenAI-compatible chat-completions endpoint for textual description generation

```bash
pip install -r requirements.txt
```

### 📥 Inputs

#### S-FFSD

Download `S-FFSD.zip` from the `data` directory of the official [AI4Risk/antifraud](https://github.com/AI4Risk/antifraud) repository. Extract the archive and place the resulting file at:

```text
data/S-FFSD.csv
```

The S-FFSD table must contain:

```text
Time, Source, Target, Amount, Location, Type, Labels
```

#### IEEE-CIS

Download the training data from the official [IEEE-CIS Fraud Detection competition data page](https://www.kaggle.com/competitions/ieee-fraud-detection/data). Place the downloaded files at:

```text
data/ieee/train_transaction.csv
data/ieee/train_identity.csv
```

IEEE-CIS provides temporal references for intraday, weekly, and calendar transaction patterns used in timestamp synthesis.

#### External Knowledge Base

The repository already includes the statistical reference tables and semantic priors used by the agents during initialization and consistency refinement. 

```text
initialization/informations/
consistency_optimization/priors/
```

### ▶️ Run

Run commands from the repository root:

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
python run_pipeline.py all   --base-url "$LLM_BASE_URL"   --api-key "$LLM_API_KEY"   --model "$LLM_MODEL"
```

The textual generation stage uses deterministic templates for single-transaction users and calls the LLM only for multi-transaction users and merchants. Interrupted LLM generation resumes from checkpoints in `work/checkpoints/`.

### 📦 Publishing and Validation

`python run_pipeline.py publish` validates all foreign keys, row counts, required values, transaction ordering, immutable transaction fields, and English-only public text before creating:

```text
dataset/MS-FFSD.zip
```


<a id="acknowledgements"></a>

## 🙏 Acknowledgements

MS-FFSD is built upon the publicly available S-FFSD dataset. IEEE-CIS Fraud Detection data are used as an external temporal reference during timestamp synthesis.

Please follow the original licenses and terms of use of the corresponding external data sources.


<a id="contributing"></a>

## 🤝 Contributing

Contributions are welcome. We encourage using this framework to semantically enrich financial transaction data and contribute to building more open multimodal financial datasets.

<a id="citation"></a>

## 📚 Citation

If you find this work or the MS-FFSD dataset useful in your research, please consider citing our paper:

```bibtex
@misc{shao2026behaviorgroundedsemanticenrichmentfinancial,
  title={Behavior-Grounded Semantic Enrichment for Financial Fraud Modeling and Reasoning},
  author={Linbo Shao and Huilin He and Yating Lou and Dawei Cheng},
  year={2026},
  eprint={2609.34211},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2609.34211}
}
```
