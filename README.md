# DRL_Demo
DRL HQ Pharma Intelligence Console

A Streamlit app for semantic search and grounded question answering over a pharmaceutical drug catalog stored in Couchbase. The app uses embedding-based retrieval plus LLM summarization to answer catalog questions with record-backed responses.
Features

    Semantic search over drug catalog records

    Couchbase vector retrieval using APPROX_VECTOR_DISTANCE

    Grounded answer generation with strict prompt guardrails

    Pharma-themed Streamlit UI

    Environment-variable based configuration for safer git usage

Project structure

text
.
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── run.sh

Prerequisites

    Python 3.10, 3.11, or 3.12

    pip

    Access to your Couchbase Capella cluster

    Valid embedding and chat API keys

Quick start
1) Clone the repo

bash
git clone <your-repo-url>
cd pharma-search-app

2) Create and activate a virtual environment
macOS / Linux

bash
python3 -m venv .venv
source .venv/bin/activate

Windows PowerShell

powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

3) Install dependencies

bash
pip install --upgrade pip
pip install -r requirements.txt

4) Configure environment variables

Copy the example file and update it with your real credentials.
macOS / Linux

bash
cp .env.example .env

Windows PowerShell

powershell
Copy-Item .env.example .env

Then set the values from .env in your shell, or export them manually.
macOS / Linux

bash
export CB_CONN_STR="couchbases://your-cluster"
export CB_USERNAME="your_username"
export CB_PASSWORD="your_password"
export CB_BUCKET="drl_medical"
export CB_SCOPE="catalog"
export CB_COLLECTION="drugs"
export CAPELLA_AI_BASE_URL="https://your-ai-endpoint/v1"
export EMBEDDING_MODEL="nvidia/llama-3.2-nv-embedqa-1b-v2"
export EMBEDDING_API_KEY="your_embedding_api_key"
export CHAT_MODEL="mistralai/mistral-7b-instruct-v0.3"
export CHAT_API_KEY="your_chat_api_key"
export TOP_K="5"
export VECTOR_METRIC="L2"
export NPROBES="4"

Windows PowerShell

powershell
$env:CB_CONN_STR="couchbases://your-cluster"
$env:CB_USERNAME="your_username"
$env:CB_PASSWORD="your_password"
$env:CB_BUCKET="drl_medical"
$env:CB_SCOPE="catalog"
$env:CB_COLLECTION="drugs"
$env:CAPELLA_AI_BASE_URL="https://your-ai-endpoint/v1"
$env:EMBEDDING_MODEL="nvidia/llama-3.2-nv-embedqa-1b-v2"
$env:EMBEDDING_API_KEY="your_embedding_api_key"
$env:CHAT_MODEL="mistralai/mistral-7b-instruct-v0.3"
$env:CHAT_API_KEY="your_chat_api_key"
$env:TOP_K="5"
$env:VECTOR_METRIC="L2"
$env:NPROBES="4"

5) Run the app

bash
streamlit run app.py

Or use:

bash
bash run.sh

Example prompts

    Give me Respiratory-class drugs in Phase III with side effects

    Find approved endocrine drugs with pregnancy category A

    Show controlled substance drugs in oncology

    Give me a respiratory drug with storage and distribution details

Notes

    Do not commit real secrets to git.

    The app validates required environment variables at startup.

    If Couchbase installation fails on uncommon platforms, use a platform with official wheels or install the system build dependencies recommended by Couchbase.

Troubleshooting
Couchbase install issues

The official Couchbase Python SDK may require native build tools on platforms where wheels are unavailable. Couchbase documents that some environments may need a C++ compiler and Python development files. Refer to Couchbase installation docs for platform-specific guidance.
Connection issues

If app startup fails:

    Verify DNS and outbound network access

    Check Capella TLS connectivity

    Confirm bucket, scope, and collection names

    Verify API keys and base URL

Git hygiene

Before pushing:

    Keep .env out of git

    Rotate any keys that were previously exposed

    Review commit history for accidental secrets

License

Use and adapt for your internal demo or project needs.
