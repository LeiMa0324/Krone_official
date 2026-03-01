# KRONE: Hierarchical and Modular Log Anomaly Detection (Accepted at ICDE 2026)

[![arXiv](https://img.shields.io/badge/arXiv-2602.07303-b31b1b.svg)](https://arxiv.org/abs/2602.07303)
[![Conference](https://img.shields.io/badge/ICDE-2026-blue.svg)](https://ieee-icde.org/)

> **KRONE: Hierarchical and Modular Log Anomaly Detection**
>
> Accepted at **ICDE 2026** (IEEE International Conference on Data Engineering)
>
> Lei Ma, Jinyang Liu, Tieying Zhang, Peter M. VanNostrand, Dennis M. Hofmann, Lei Cao, Elke A. Rundensteiner, Jianjun Chen

## ✨ Highlights

- 🌳 **Hierarchical Execution Recovery**: LLM automatically derives execution hierarchies (entity, action, status) from log templates, decomposing flat log sequence into coherent execution chunks (KRONE-Seqs) at entity, action, and status levels.
- 🔍 **Modular Multi-Level Detection**: Performs targeted anomaly identification at each semantic level, enabling precise localization of *where* and *why* an anomaly occurs.
- ⚡ **Hybrid Detection Strategy**: Dynamically routes between efficient local pattern matching filtering and LLM-powered nested-aware detection, reducing LLM usage to only a small fraction of test data. 🔌 **KRONE is detector-agnostic — plug in any log anomaly detector and benefit from the hierarchy. Contributions & extensions are welcome!**
- 🏆 **State-of-the-Art Performance**:  Experiments on three public benchmarks and one industrial dataset from ByteDance Cloud demonstrate the comprehensive improvement of KRONE,  F-1 of same detector with or without hierarchy), dataefficiency (data space 117.3× ↓), resource-efficieny (43.7× ↓) and
interpretability. KRONE improves F1-score by 10.07% (82.76% → 92.83%) over prior methods, while reducing **LLM usage to 1.1%–3.3% of the test data size**.

## 📖 Overview

Logs originate from nested component executions with clear structural boundaries, but this organization is lost when stored as flat sequences. KRONE recovers this structure by constructing a hierarchical Log Abstraction Model and performing modular anomaly detection at three abstraction levels:

```
ROOT → ENTITY → ACTION → STATUS
```

- **Entity Level**: System modules or components (e.g., `PacketResponder`, `block`)
- **Action Level**: Operations performed (e.g., `creating`, `receiving`, `terminating`)
- **Status Level**: Outcomes (e.g., `success`, `failure`, `exception`)

## 🚀 Quick Start

We provide demo sampled datasets (~20K sequences each) under `data/` for quick experimentation. For full-scale datasets used in the paper, please refer to the original sources.

### 📦 Prerequisites

```bash
pip: pip install pandas numpy scikit-learn sentence-transformers openai tqdm torch python-dotenv
conda: conda env create -f environment.yml
```

### 📊 Built-in Datasets

| Dataset | Domain | Description |
|---------|--------|-------------|
| **BGL** | Supercomputing | Blue Gene/L supercomputer logs |
| **HDFS** | Distributed Systems | Hadoop Distributed File System logs |
| **Thunderbird** | Supercomputing | Thunderbird system event logs |


### 🌳 Step 1: KRONE-Tree Extraction from Log Templates (LLM Required)

Extract the hierarchical KRONE-Tree structure from raw log templates using `tree_extraction/extractor.py`. The extracted tree is saved to `output/{dataset}/templates_krone_tree.csv`.

```bash
python tree_extraction/extractor.py
```

> We have included pre-extracted KRONE-Trees in the repo, so you can skip this step and go directly to Step 2.

### 🔬 Step 2: Run Detection

#### Detection Modes

| Mode | Description | LLM Required |
|------|-------------|:---:|
| `local` | Automaton-based pattern matching — fast and efficient | No |
| `mix` | Hybrid: local filtering + LLM on a subset — balanced | Yes |


For local detection using pattern matching, simply run a demo script with the default config:

```bash
cd demo_main
python BGL.py
python HDFS.py
python ThunderBird.py
```

Expected results on the demo sampled datasets with default local detection config:

| Dataset         | F1 | Precision | Recall | TP | FP | TN | FN |
|-----------------|------|-----------|--------|------|------|------|------|
| **HDFS**        | 0.9838 | 0.9698 | 0.9983 | 578 | 18 | 3403 | 1 |
| **BGL**         | 0.9766 | 0.9542 | 1.0000 | 1835 | 88 | 2077 | 0 |
| **ThunderBird** | 0.8368 | 0.7195 | 1.0000 | 159 | 62 | 3779 | 0 |


For LLM integrated detection, first set your OpenAI API keys in the `.env` file:
and then choose the LLM config predefined in the above demo_main scripts:



## 🗂️ Project Structure

```
KRONE_official/
├── demo_main/               # Entry points for each dataset
│   ├── BGL.py
│   ├── HDFS.py
│   └── Thunderbird.py
├── executor/                # Pipeline orchestration
│   └── executor.py
├── krone_hierarchy/         # Core krone execution orchestration
│   ├── Krone_tree.py        # Hierarchical Krone tree construction
│   ├── Node.py              # Node data structure
│   ├── Krone_seq.py         # KRONE-Seqs representation ()
│   ├── Krone_seq_manager.py # KRONE-Seqs management & automaton
│   ├── KnowledgeBase.py     # Knowledge base for KRONE-Seq, emebdding, summary, and LLM detection result, explanation
│   ├── Automaton_graph.py   # State machine optional detector
│   ├── PROMPTS.py           # LLM prompts
├── tree_extraction/         # Krone-tree extraction (LLM)
│   ├── extractor.py
│   └── EXTRACT_PROMPTS.py
├── llm/                    # LLM configuration (OpenAI API)
│   └── llm.py
├── data/                   # Datasets
│   ├── BGL/
│   ├── HDFS/
│   └── ThunderBird/
├── output/                 # Results & Knowledge base content
└── utils.py                # Metrics (AUC, F1, precision, recall)
```



## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{ma2026kronehierarchicalmodularlog,
      title={KRONE: Hierarchical and Modular Log Anomaly Detection}, 
      author={Lei Ma and Jinyang Liu and Tieying Zhang and Peter M. VanNostrand and Dennis M. Hofmann and Lei Cao and Elke A. Rundensteiner and Jianjun Chen},
      year={2026},
      eprint={2602.07303},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2602.07303}, 
}
```
