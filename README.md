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

    Python 3.13 and above

    pip

    Access to your Couchbase Capella cluster

    Valid embedding and chat API keys

Quick start
1) Clone the repo

bash
git clone <your-repo-url>
cd DRL_Demo

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

4) Run the app

bash
streamlit run v2_vector_app.py

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
