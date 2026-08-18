# AI Tutor Evaluation Toolkit

![Toolkit Overview](./assets/others/aitutor-evalkit-main.png)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Standalone paper-best LoMTL API

This repository includes a standalone FastAPI service for the **paper-best LoMTL automated tutor evaluator only**. It keeps the original `EvaluationDatasetFormatter` prompt, task and label definitions, Gemma chat template, and deterministic generation (`do_sample=False`). The fixed runtime is:

- Base model: `google/gemma-2-2b-it`
- LoRA checkpoint: `assets/model/lora_model`
- Dimensions: `Mistake_Identification`, `Mistake_Location`, `Providing_Guidance`, and `Actionability`
- Labels: `Yes`, `To some extent`, or `No`

The model and LoRA adapter load once during server startup and remain resident on the GPU. API callers cannot select another model, checkpoint, task, or generation configuration.

### End-to-end setup: installation to first request

The following is the complete recommended workflow for the configured machine. Commands marked **first time only** do not need to be repeated for every API start.

#### 1. Clone and enter the repository (first time only)

```bash
git clone https://github.com/teinhonglo/AITutor-EvalKit.git
cd AITutor-EvalKit
```

If the repository is already cloned, only run `cd /path/to/AITutor-EvalKit`.

#### 2. Initialize Conda and create the environment (first time only)

```bash
eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda env create -f environment.yml
conda activate teval_py310
```

If `teval_py310` already exists, update it instead of creating it again:

```bash
conda env update -n teval_py310 -f environment.yml --prune
conda activate teval_py310
```

Confirm that the API and model packages import successfully:

```bash
python -c "import fastapi, peft, torch, transformers, uvicorn; print('dependencies: ok')"
```

#### 3. Confirm GPU and checkpoint availability

```bash
nvidia-smi
test -f assets/model/lora_model/adapter_model.safetensors \
  && echo "LoRA checkpoint: ok"
```

The API requires a CUDA-visible GPU. The committed adapter must remain at `assets/model/lora_model`.

#### 4. Authenticate with Hugging Face (first time only per account/machine)

Gemma may require accepting its license on the `google/gemma-2-2b-it` Hugging Face model page. After access is approved, log in with one of these methods:

```bash
hf auth login
```

or set a token for the current shell:

```bash
export HF_TOKEN=hf_your_token_here
```

Do not commit the token to this repository.

#### 5. Start the API and random Cloudflare URL

```bash
bash run_api.sh
```

`run_api.sh` automatically sources `path.sh`, activates `teval_py310`, selects GPU 0, loads Gemma and the LoRA once, waits for `/health`, creates a Cloudflare Quick Tunnel, and finally prints the local and randomly generated public URLs. Initial startup can take several minutes while Hugging Face downloads the base model.

To use GPU 1 and port 8080 instead:

```bash
GPU_ID=1 bash run_api.sh --port 8080
```

Keep this terminal open. Do not copy a Cloudflare URL until the launcher displays `Cloudflare random public URL (ready)`.

#### 6. Test locally from a second terminal

For the default port, open another terminal in the repository and run:

```bash
eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate teval_py310
python scripts/test_api.py --base-url http://127.0.0.1:8000
```

If startup used `--port 8080`, replace `8000` with `8080`. A direct request can also be sent without the Python client:

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d @tests/sample_request.json
```

#### 7. Test the public Cloudflare URL

Copy the exact random URL printed by `run_api.sh`; do not use the placeholder literally:

```bash
python scripts/test_api.py \
  --base-url https://the-actual-random-name.trycloudflare.com
```

This command can be run from another machine after installing Python and `requests`. Public Swagger documentation is available by appending `/docs` to the printed URL.

#### 8. Stop everything

Return to the terminal running `run_api.sh` and press:

```text
Ctrl+C
```

The launcher's cleanup trap stops both Uvicorn and `cloudflared`. A Quick Tunnel URL is temporary; the next startup normally produces a different random URL.

#### Optional: local-only startup

To skip Cloudflare entirely:

```bash
bash run_api.sh --no-tunnel
```

### Install

A CUDA-capable Linux machine, Python 3.10+, Git, and `curl` are required. Gemma is a gated Hugging Face model: accept its license on Hugging Face, then authenticate before startup (for example, `huggingface-cli login` or export `HF_TOKEN`).

Use the existing Conda environment (which includes the API dependencies):

```bash
conda env create -f environment.yml
conda activate teval_py310
```

For one-command startup, `run_api.sh` automatically sources the root `path.sh` before checking dependencies or starting Uvicorn. `path.sh` runs the requested Conda initialization and activates `teval_py310`:

```bash
eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate teval_py310
```

If Conda is installed at a different location or a different environment name is needed, override either value without editing the scripts:

```bash
CONDA_BIN=/opt/conda/bin/conda CONDA_ENV_NAME=teval_py310 bash run_api.sh
```

Alternatively, install the normalized, unambiguous root requirements file:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for your system if the default package does not provide CUDA support. The API intentionally fails startup when no visible CUDA device is available.

### One-command startup

```bash
bash run_api.sh
```

This command first initializes Conda through `path.sh`, then defaults to GPU 0, loads Gemma and the committed LoRA adapter, starts FastAPI, waits until `/health` confirms model readiness, launches a free Cloudflare Quick Tunnel, and prints both local and public URLs. It uses `cloudflared` from `PATH` when present; otherwise it downloads the official binary into the ignored `.runtime/` directory without `sudo`.

Select another physical GPU before Python starts:

```bash
GPU_ID=1 bash run_api.sh
```

Run without downloading or launching Cloudflare:

```bash
bash run_api.sh --no-tunnel
```

Choose the API port explicitly with `--port` (this affects both the local API and the Cloudflare tunnel origin):

```bash
bash run_api.sh --port 8080
```

Options can be combined:

```bash
GPU_ID=1 bash run_api.sh --port 8080 --no-tunnel
```

The `PORT=8080` environment-variable form remains supported. `HOST` and the model startup timeout can also be adjusted (for example, `STARTUP_TIMEOUT=900 bash run_api.sh --port 8080`). Keep `HOST=127.0.0.1` for the Quick Tunnel's local origin.

Cloudflare Quick Tunnel generates a **new random `https://*.trycloudflare.com` URL**. The launcher waits until that URL has actually been generated and then prints it prominently, together with its Swagger URL and a ready-to-copy remote test command. No URL is hard-coded or printed before the tunnel is ready.

### API schema

#### `GET /health`

A successful response is returned only after model initialization completes:

```json
{
  "status": "ok",
  "model_ready": true,
  "model": "google/gemma-2-2b-it",
  "evaluator": "LoMTL"
}
```

#### `POST /evaluate`

Request (additional fields are rejected):

```json
{
  "conversation_history": "Tutor: We are rounding 1551. Student: It rounds to 1500.",
  "tutor_response": "Check the tens digit: it is 5, so round up to 1600."
}
```

Response:

```json
{
  "Mistake_Identification": "Yes",
  "Mistake_Location": "To some extent",
  "Providing_Guidance": "Yes",
  "Actionability": "Yes"
}
```

An invalid generated label is never coerced: the server returns HTTP 502 with an explicit explanation. Empty or unexpected request fields receive FastAPI's HTTP 422 validation response, unavailable model state receives HTTP 503, and other inference failures receive HTTP 500.

Interactive OpenAPI documentation is available at `/docs` on either URL.

### Test locally or remotely

The client first prints and sends the representative request in `tests/sample_request.json`, checks `/health`, then validates all four returned labels. This is a real example adapted from the first conversation and its `GPT4` tutor response in `assets/data/test_data/test_sample.json`, rather than a fabricated API-only example. Test the local service:

```bash
python scripts/test_api.py \
  --base-url http://127.0.0.1:8000
```

Copy the generated URL printed by `run_api.sh` to test through Cloudflare from this or another machine:

```bash
python scripts/test_api.py \
  --base-url https://xxxxxxxx.trycloudflare.com
```

The client does not assume localhost; any reachable API base URL is accepted. A direct curl request works the same way:

```bash
curl -X POST \
  http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d @tests/sample_request.json
```

Replace the local base URL with the printed `https://*.trycloudflare.com` URL for a public request. Press **Ctrl+C** in the `run_api.sh` terminal to terminate both Uvicorn and Cloudflare; the script also performs the same cleanup on SIGTERM.

### Basic functional tests (no GPU required)

The FastAPI lifecycle and HTTP behavior can be tested without downloading Gemma or using CUDA. These tests inject a lightweight evaluator through the same application lifecycle and verify health, evaluation, strict request validation, error responses, and that the evaluator is initialized only once:

```bash
python -m pytest -q tests/test_api_app.py
```

This CPU-only test does not claim to validate model quality. After it passes, use `scripts/test_api.py` against a running CUDA service for the end-to-end Gemma + LoRA smoke test described above.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#-quick-start)
  - [Backend Module: AI Tutor Evaluation](#-backend-module-ai-tutor-evaluation-module)
    - [Evaluation with Automated Models](#evaluation-with-automated-model)
    - [Evaluation with Open-Source Models](#evaluation-with-open-source-model)
    - [Evaluation with GPT Models](#evaluation-with-gpt-5-model)
  - [Frontend Module: Demo App](#frontend-module-demo-app)
    - [Start the Local Server](#2-start-the-local-server)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Overview
Welcome to **`AITutor-EvalKit`** ! 

**`AITutor-EvalKit`** provides a robust framework for assessing the **pedagogical effectiveness** of large language models (LLMs) as AI tutors in the **mathematics** domain.  

Developed by the **EduNLP Lab at MBZUAI**, this toolkit combines insights from **learning sciences** with recent advancements in **LLM technology** to systematically **evaluate**, **compare**, and **analyze** the pedagogical performance of AI tutoring models. Building on prior research – including the **NAACL 2025 SAC Award-winning paper** by *Maurya et al.* [1] and the **BEA 2025 Shared Task** by *Kochmar et al.* [2] – it introduces a **lightweight, modular framework** for **on-the-fly evaluation** of AI tutor responses, extending beyond static benchmarks to enable scalable and principled pedagogical assessment.

This repository provides comprehensive information and usage instructions for the two main modules of the toolkit: **Backend** and **Frontend**. The backend includes multiple evaluation options, ranging from automated evaluation models to various LLM-as-a-judges. The frontend seamlessly connects the evaluation models with the demo app, enabling users to interactively explore the toolkit’s effectiveness and capabilities.

**Key Features:**
- **Modular Design** - Use modules independently or jointly
- **Customizable** - Easy prompt and configuration customization
- **Multi-dimensional** - Evaluate across multiple dimensions
- **Cost-Effective** - LoRA training uses 10x less memory than full fine-tuning
- **Local App** - Qucikly launch the local App with evaluations/annotations
- **Production-Ready** - Robust error handling and logging

---

## Project Resources

Below are the key resources for the AITutor-EvalKit project:

<table>
  <tr><td><strong>Paper:</strong> <a href="https://arxiv.org/abs/2512.03688" target="_blank">https://arxiv.org/abs/2512.03688</a></td></tr>
  <tr><td><strong>Toolkit App:</strong> <a href="https://demo-ai-tutor.vercel.app/" target="_blank">https://demo-ai-tutor.vercel.app/</a></td></tr>
  <tr><td><strong>Demo Video:</strong> <a href="https://www.youtube.com/watch?v=9qgDfrhzOvg" target="_blank">https://www.youtube.com/watch?v=9qgDfrhzOvg</a></td></tr>
  <tr><td><strong>Associated Data Source I:</strong> <a href="https://aclanthology.org/2025.naacl-long.57/" target="_blank">https://aclanthology.org/2025.naacl-long.57/</a></td></tr>
  <tr><td><strong>Associated Data Source II:</strong> <a href="https://aclanthology.org/2025.bea-1.77/" target="_blank">https://aclanthology.org/2025.bea-1.77/</a></td></tr>
</table>

---


## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- A minimum of 48 GB single GPU is recommended for model training and evaluation
- HuggingFace account & CLI login (export HuggingFace token in the environment to access gated models)
- Tested with Ubuntu 24.04.1 LTS

### Step 1: Clone Repository

```bash
git clone https://github.com/teinhonglo/AITutor-EvalKit.git
cd AITutor-EvalKit
```

---

### **Step 2. Verify the Directory Structure**

Ensure your cloned repository has the following structure:

```
AITutor-EvalKit/
        │
        ├── README.md                      # This file
        ├── environment.yml                # Dependencies
        │
        ├── src/                          # Main repository for the AI tutor evaluation 
        |   ├── README.md                 # AutoEval and LLMEval Documentation
        │   ├── autoeval/                 # LoRA Training & Evaluation
        │   │   ├── train.py              # Training script
        │   │   ├── evaluation.py         # Evaluation script
        │   │   ├── lora_finetune_runner.sh
        │   │   ├── lora_evaluation_runner.sh
        │   │   └── utils/
        │   │
        │    llmeval/                     # LLM-as-a-Judge
        │       ├── gpt5_eval.py          # OpenAI GPT evaluation
        │       ├── gpt5_eval_runner.sh   
        │       ├── open_llmeval.py       # Open-source LLM evaluation
        │       ├── run_open_llm_as_judge_evaluation.py
        │       └── utils
        │
        ├── app_src/                      # Main repository for tool app
        │   ├── README.md                 # API Documentation
        |   |── api/                       
        │   ├── scripts/
        │   ├── ....
        │   └── others/
        |
        └── assets/
            ├── data/
            │   ├── train_data/           # Training datasets
            │   └── test_data/            # Test datasets
            │
            ├── model/                    # LoRa checkpoint
            ├── outputs/                  # Evaluated output files
            └── others/                   # Other assests
```

If any of these directories are missing, create them before proceeding.

---

### **Step 3. Create the Conda Environment**

Ensure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

Create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate teval_py310
```

---

### **Step 4. Prepare the Dataset and Train or Download Evaluation Models**

> Place all dataset files in the `assets/data/` directory and the automated evaluation model in the `assets/model/` directory.
> Ensure that test data is available at `assets/data/test_data/test_sample.json` and the model is located at `assets/model/lora_model`.
> For further information, please refer to the `src/README.md` file.

**Example structure:**

```
data/
├── test_data
|      └── test_sample.json
└── train_data/
```

```
model/
└── lora_model/
```

Ensure that all dataset and model paths are correctly referenced in scripts or configuration files.

---

### **Sample Test Example**

```json
{
   "conversation_id": "01-374a3eb6-95cf-4725-9e76-86a8972aa5cb",
    "conversation_history": "Tutor:  Hi, could you please provide a step-by-step solution for the question below? The question is: ...",
    "Data": "Not Available",
    "Split": "Not Available",
    "Topic": "Not Available",
    "Problem_topic": "Object Counting Problem",
    "Ground_Truth_Solution": "The total number of spoons from Julia and her husband was 12+3=15 spoons...",
    "anno_llm_responses": {
      "Gemini": {
        "response": "That's great! Now, remember her husband gave her 5 spoons, so how many did she have *before* that?"
      },
      "Phi3": {
        "response": "Great job! Now let's try solving a similar problem together."
      },
      "Llama-3.1-8B": {
        "response": "That's correct, Julia had 9 spoons left, but let's not forget that her husband also bought ,,,"
      },
    ...
  }
}
```

A sample test file with 10 examples is available at: `assets/data/test_data/test_sample.json`

---

## Quick Start

### Backend Module: AI Tutor Evaluation Module

All backend-related code is located in the [`src/`](src/) directory.  
For a detailed overview of backend components and configurations, refer to the [Backend README](src/README.md).

#### **Evaluation with an Automated Model**

Run the following command to perform automated evaluation using the our LoRA model:

```bash
bash src/autoeval/lora_evaluation_runner.sh
```

#### **Evaluation with an Open-Source Model**

To evaluate using an open-source model, execute:

```bash
python src/llmeval/run_open_llm_as_judge_evaluation.py
```

#### **Evaluation with the GPT-5 Model**

To perform evaluation with the GPT-5 model, run:

```bash
bash src/llmeval/gpt5_eval_runner.sh
```

After completing the evaluation stage, the output file should follow the structure shown below:


**Sample evaluated/annotated example**

```json
{
  "conversation_id": "01-374a3eb6-95cf-4725-9e76-86a8972aa5cb",
  "conversation_history": "Tutor: Hi, could you please provide a step-by-step solution for the question below? The question is: ...",
  "Data": "Not Available",
  "Split": "Not Available",
  "Topic": "Not Available",
  "Problem_topic": "Object Counting Problem",
  "ground_truth_solution": "The total number of spoons from Julia and her husband was 12 + 3 = 15 spoons ..",
  "anno_llm_responses": {
    "Gemini": {
      "response": "That's great! Now, remember her husband gave her 5 spoons, so how many did she have *before* that?",
      "auto_annotation": {
        "Mistake_Identification": "Yes",
        "Mistake_Location": "Yes",
        "Providing_Guidance": "Yes",
        "Actionability": "Yes"
      },
      "llm_annotation": {
        "Mistake_Identification/prometheus-eval/prometheus-7b-v2.0": "No",
        "Mistake_Location/prometheus-eval/prometheus-7b-v2.0": "No",
        "Providing_Guidance/prometheus-eval/prometheus-7b-v2.0": "To some extent",
        "Actionability/prometheus-eval/prometheus-7b-v2.0": "Yes",
        "Mistake_Identification/gpt5": "No",
        "Mistake_Location/gpt5": "No",
        "Providing_Guidance/gpt5": "No",
        "Actionability/gpt5": "Yes"
      }
    },
    ...
  }
}
```

A complete sample file evaluated/annotated by the backend module is available at `assets/outputs/gpt5_model_predictions.json`. This file will be used by the frontend demo app module. 


### Frontend Module: Demo App

All frontend-related code is located in the [`app_src/`](app_src/) directory.
For implementation details and setup instructions, refer to the [Frontend README](app_src/README.md).

**1. Install Dependencies**

Install all required packages using **npm**:

```bash
npm install
```

**2. Start the Local Server**

#### **Using the Default Dataset**

```bash
npm run dev
```

#### **Using a Custom Dataset**

```bash
npm run dev:custom /path/to/your/dataset.json
```

**Example:**

```bash
npm run dev:custom assets/outputs/gpt5_model_predictions.json
```

## Troubleshooting

### **Common Issues**

**1. CUDA Out of Memory**
```bash
# Reduce batch size
BATCH_SIZE=1
GRAD_ACCUM=8

# Enable gradient checkpointing
--gradient_checkpointing
```

**2. Path Errors**
```bash
# Use absolute paths
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${WORKSPACE_ROOT}/assets/data/"
```

**3. OpenAI Rate Limits**
```python
# Adjust in gpt5_eval.py
REQUESTS_PER_MINUTE = 20  # Lower for Tier 1
```

 
## Documentation

### **Detailed Guides**
- **[Backend Module](src/README.md)** - Complete AutoEval & LLMEval documentation
- **[Frontend Module](app_src/README.md)** - Demo App setup and customization
- **[Changelog](CHANGELOG.md)** - Version history and updates

### **Quick Reference Commands**

```bash
# Training
cd src/autoeval && ./lora_finetune_runner.sh

# Evaluation (Automated)
cd src/autoeval && ./lora_evaluation_runner.sh

# Evaluation (Open LLM)
cd src/llmeval && python run_open_llm_as_judge_evaluation.py

# Evaluation (GPT)
cd src/llmeval && ./gpt5_eval_runner.sh

# Frontend Demo
cd app_src && npm run dev
```


## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Submit a pull request with clear description

---

## License

This project is licensed under the MIT License.

---

## Contact

- **Issues:** [GitHub Issues](https://github.com/teinhonglo/AITutor-EvalKit/issues)
- **Lab:** [EduNLP Lab, MBZUAI](https://mbzuai.ac.ae)

---

## Citation

If you use **AITutor-EvalKit** in your research or project, please cite our paper:

```bibtex
@misc{naeem2025aitutor,
  title         = {AITutor-EvalKit: Exploring the Capabilities of AI Tutors},
  author        = {Numaan Naeem and Kaushal Kumar Maurya and Kseniia Petukhova and Ekaterina Kochmar},
  year          = {2025},
  eprint        = {2512.03688},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2512.03688}
}
```

---

## Acknowledgments

Developed by the **EduNLP Lab at MBZUAI**

Special thanks to:
- Google for Google Academic Research Award (GARA) supporting this research
- HuggingFace for Transformers & PEFT
- OpenAI for GPT APIs
- vLLM team for efficient inference
- Meta AI, Google, Mistral for open-source models

---

## References

1. Maurya, Kaushal Kumar, KV Aditya Srivatsa, Kseniia Petukhova, and Ekaterina Kochmar. *"Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors."* In *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pp. 1234–1251. 2025.  

2. Ekaterina Kochmar, Kaushal Maurya, Kseniia Petukhova, Kv Aditya Srivatsa, Anaïs Tack, and Justin Vasselli. *"Findings of the BEA 2025 Shared Task on Pedagogical Ability Assessment of AI-Powered Tutors."* In *Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2025)*, pp. 1011–1033, Vienna, Austria. Association for Computational Linguistics. 2025.

---

**Version:** 1.0.0  
**Last Updated:** December 1, 2025
