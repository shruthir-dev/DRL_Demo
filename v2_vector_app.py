import re
import streamlit as st
from datetime import timedelta
from typing import List, Dict, Any, Tuple

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster, ClusterOptions
from couchbase.options import QueryOptions
from openai import OpenAI


CB_CONN_STR = "couchbases://cb.2ftxhxcz27u6meyb.cloud.couchbase.com"
CB_USERNAME = "demo_access"
CB_PASSWORD = "Demo@123"
CB_BUCKET = "drl_medical"
CB_SCOPE = "catalog"
CB_COLLECTION = "drugs"

CAPELLA_AI_BASE_URL = "https://xbxg79oc1emidd.ai.cloud.couchbase.com/v1"
EMBEDDING_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
EMBEDDING_API_KEY = "cbsk-v1-Mhi4NoCEjR2KNCYAhGIqqsnaS6T1qzv52Gkf4LfT19JSv0Kh"
CHAT_MODEL = "mistralai/mistral-7b-instruct-v0.3"
CHAT_API_KEY = "cbsk-v1-CUlGPIG1sAwhUXQXFPaU1tgJSfuEibqHc5lCXbhjYN74VyFU"

TOP_K = 5
VECTOR_METRIC = "L2"
NPROBES = 4


@st.cache_resource
def get_cluster():
    auth = PasswordAuthenticator(CB_USERNAME, CB_PASSWORD)
    options = ClusterOptions(auth)
    cluster = Cluster(CB_CONN_STR, options)
    cluster.wait_until_ready(timedelta(seconds=30))
    return cluster


@st.cache_resource
def get_embedding_client():
    return OpenAI(base_url=CAPELLA_AI_BASE_URL, api_key=EMBEDDING_API_KEY)


@st.cache_resource
def get_chat_client():
    return OpenAI(base_url=CAPELLA_AI_BASE_URL, api_key=CHAT_API_KEY)


def create_query_embedding(client: OpenAI, text: str) -> List[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


KNOWN_CLASSES = [
    "respiratory",
    "cardiovascular",
    "endocrine",
    "neurology",
    "oncology",
    "gastroenterology",
    "dermatology",
    "infectious disease",
    "immunology",
]

KNOWN_REG_STATUS = [
    "approved",
    "phase i",
    "phase ii",
    "phase iii",
    "phase iv",
    "withdrawn",
    "under review",
]


def detect_filters(question: str) -> Tuple[List[str], Dict[str, Any]]:
    q = question.lower()
    where_clauses = ["doc.medical_vector IS NOT NULL"]
    params: Dict[str, Any] = {}

    for c in KNOWN_CLASSES:
        if c in q:
            where_clauses.append("LOWER(doc.therapeutic_class) = $therapeutic_class")
            params["therapeutic_class"] = c
            break

    for r in KNOWN_REG_STATUS:
        if r in q:
            where_clauses.append("LOWER(doc.regulatory_status) = $regulatory_status")
            params["regulatory_status"] = r
            break

    preg_match = re.search(r'pregnancy category\s*([a-z])', q)
    if preg_match:
        where_clauses.append("LOWER(doc.clinical_info.pregnancy_category) = $pregnancy_category")
        params["pregnancy_category"] = preg_match.group(1).lower()

    if "controlled substance" in q:
        if "not" in q or "non" in q:
            where_clauses.append("doc.supply_chain.is_controlled_substance = FALSE")
        else:
            where_clauses.append("doc.supply_chain.is_controlled_substance = TRUE")

    return where_clauses, params


def retrieve_documents(cluster: Cluster, query_vector: List[float], question: str, limit: int = TOP_K) -> List[Dict[str, Any]]:
    keyspace = f"`{CB_BUCKET}`.`{CB_SCOPE}`.`{CB_COLLECTION}`"
    where_clauses, extra_params = detect_filters(question)
    where_sql = " AND ".join(where_clauses)

    sql = f"""
    SELECT
        META(doc).id AS doc_id,
        doc.drug_id AS drug_id,
        doc.generic_name AS generic_name,
        doc.brand_name AS brand_name,
        doc.manufacturer AS manufacturer,
        doc.therapeutic_class AS therapeutic_class,
        doc.regulatory_status AS regulatory_status,
        doc.approval_year AS approval_year,
        doc.active_ingredient_percentage AS active_ingredient_percentage,
        doc.clinical_info.primary_indication AS primary_indication,
        doc.clinical_info.mechanism_of_action AS mechanism_of_action,
        doc.clinical_info.pregnancy_category AS pregnancy_category,
        doc.clinical_info.contraindications AS contraindications,
        doc.adverse_reactions.common AS common_adverse_reactions,
        doc.adverse_reactions.severe AS severe_adverse_reactions,
        doc.specifications.dosage_form AS dosage_form,
        doc.specifications.strength AS strength,
        doc.specifications.shelf_life_months AS shelf_life_months,
        doc.specifications.storage_conditions AS storage_conditions,
        doc.supply_chain.unit_price_usd AS unit_price_usd,
        doc.supply_chain.is_controlled_substance AS is_controlled_substance,
        doc.supply_chain.distribution_regions AS distribution_regions,
        APPROX_VECTOR_DISTANCE(doc.medical_vector, $query_vec, "{VECTOR_METRIC}", {NPROBES}) AS distance
    FROM {keyspace} AS doc
    WHERE {where_sql}
    ORDER BY distance
    LIMIT $limit;
    """

    params = {"query_vec": query_vector, "limit": limit}
    params.update(extra_params)

    result = cluster.query(sql, QueryOptions(named_parameters=params))
    return [row for row in result]


def build_context(rows: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, row in enumerate(rows, start=1):
        contraindications = ", ".join(row.get("contraindications") or []) or "None listed"
        common_ae = ", ".join(row.get("common_adverse_reactions") or []) or "None listed"
        severe_ae = ", ".join(row.get("severe_adverse_reactions") or []) or "None listed"
        regions = ", ".join(row.get("distribution_regions") or []) or "Not specified"

        block = f"""
[RECORD {i} | doc_id: {row.get("doc_id", "N/A")} | brand: {row.get("brand_name", "N/A")}]
Generic Name          : {row.get("generic_name", "N/A")}
Brand Name            : {row.get("brand_name", "N/A")}
Manufacturer          : {row.get("manufacturer", "N/A")}
Therapeutic Class     : {row.get("therapeutic_class", "N/A")}
Regulatory Status     : {row.get("regulatory_status", "N/A")}
Approval Year         : {row.get("approval_year", "N/A")}
Active Ingredient %   : {row.get("active_ingredient_percentage", "N/A")}
Primary Indication    : {row.get("primary_indication", "N/A")}
Mechanism of Action   : {row.get("mechanism_of_action", "N/A")}
Pregnancy Category    : {row.get("pregnancy_category", "N/A")}
Contraindications     : {contraindications}
Common Side Effects   : {common_ae}
Severe Side Effects   : {severe_ae}
Dosage Form           : {row.get("dosage_form", "N/A")}
Strength              : {row.get("strength", "N/A")}
Storage               : {row.get("storage_conditions", "N/A")}
Shelf Life            : {row.get("shelf_life_months", "N/A")} months
Unit Price (USD)      : {row.get("unit_price_usd", "N/A")}
Controlled Substance  : {row.get("is_controlled_substance", "N/A")}
Distribution Regions  : {regions}
Relevance Distance    : {round(row.get("distance", 0), 4)}
""".strip()
        blocks.append(block)

    return "\n\n".join(blocks)


SYSTEM_PROMPT = """
You are a precise clinical retrieval assistant for a pharmaceutical drug catalog.

STRICT RULES:
1. Use ONLY the fields explicitly present in the retrieved records.
2. Do NOT use outside medical knowledge.
3. Every sentence must be traceable to a specific retrieved record.
4. If records vary, say they vary by record or brand.
5. Do NOT generalize beyond the retrieved records.
6. End with a Sources Used section listing only records actually used.

RESPONSE FORMAT:
Use only relevant sections.

**Drug Overview**
...

**Side Effects**
...

**Clinical Details**
...

**Supply & Availability**
...

**Sources Used**
- [doc_id] | [Brand Name] | [Manufacturer]
"""


def generate_answer(chat_client: OpenAI, question: str, context: str) -> str:
    response = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"User question:\n{question}\n\nRetrieved catalog records:\n{context}"}
        ],
    )
    return response.choices[0].message.content.strip()


def answer_with_guardrails(chat_client: OpenAI, question: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return (
            "**No matching records found.**\n\n"
            "The retrieved catalog records do not contain sufficient information to answer this question. "
            "Try rephrasing or use a broader query."
        )
    return generate_answer(chat_client, question, build_context(rows))


def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(0, 180, 216, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(0, 119, 182, 0.12), transparent 24%),
            linear-gradient(180deg, #f4fbff 0%, #eef8f3 100%);
        color: #12324a;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1080px;
    }

    .hero {
        background: linear-gradient(135deg, #0b3c5d 0%, #0f6b78 52%, #12a39a 100%);
        border-radius: 24px;
        padding: 28px 30px;
        color: white;
        box-shadow: 0 20px 50px rgba(10, 60, 93, 0.22);
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }

    .hero-sub {
        font-size: 1rem;
        opacity: 0.92;
        line-height: 1.6;
        max-width: 760px;
    }

    .pill-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 16px;
    }

    .pill {
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.18);
        color: #ffffff;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 600;
    }

    .metric-card {
        background: rgba(255,255,255,0.78);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(16, 83, 114, 0.10);
        border-radius: 20px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 10px 30px rgba(20, 52, 74, 0.08);
        margin-bottom: 1rem;
    }

    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #4d7189;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0b3c5d;
    }

    .metric-sub {
        margin-top: 6px;
        color: #53758c;
        font-size: 0.9rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 0.4rem 0.4rem;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(180deg, rgba(18,163,154,0.10), rgba(18,163,154,0.06));
        border: 1px solid rgba(18,163,154,0.14);
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(11,60,93,0.08);
        box-shadow: 0 12px 30px rgba(8, 40, 64, 0.05);
    }

    .stChatInput > div {
        border-radius: 18px !important;
        border: 2px solid rgba(18, 163, 154, 0.28) !important;
        background: rgba(255,255,255,0.92) !important;
        box-shadow: 0 10px 24px rgba(15, 75, 102, 0.08);
    }

    .stChatInput textarea {
        color: #12324a !important;
        font-size: 1rem !important;
    }

    .stChatInput textarea::placeholder {
        color: #6e8ca1 !important;
    }

    .stExpander {
        border: 1px solid rgba(11,60,93,0.10) !important;
        border-radius: 18px !important;
        overflow: hidden;
        background: rgba(255,255,255,0.75);
    }

    .stExpander summary {
        font-weight: 700;
        color: #0b3c5d !important;
    }

    .record-card {
        background: linear-gradient(180deg, #ffffff, #f7fcff);
        border: 1px solid rgba(11,60,93,0.08);
        border-left: 6px solid #12a39a;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 8px 24px rgba(19, 53, 74, 0.05);
    }

    .record-title {
        font-size: 1rem;
        font-weight: 800;
        color: #0b3c5d;
        margin-bottom: 4px;
    }

    .record-meta {
        color: #5b7a8d;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .footer-note {
        margin-top: 1.2rem;
        padding: 14px 16px;
        border-radius: 16px;
        background: rgba(11,60,93,0.06);
        border: 1px dashed rgba(11,60,93,0.18);
        color: #32556c;
        font-size: 0.92rem;
    }

    .stAlert {
        border-radius: 16px;
    }

    h1, h2, h3 {
        color: #0b3c5d;
    }
    </style>
    """, unsafe_allow_html=True)


st.set_page_config(
    page_title="DRL HQ | Pharma Intelligence Console",
    page_icon="💊",
    layout="wide"
)

inject_custom_css()

st.markdown("""
<div class="hero">
    <div class="hero-title">💊 DRL HQ | Pharma Intelligence Console</div>
    <div class="hero-sub">
        Search clinical catalog records with grounded retrieval, vector similarity, and structured LLM summarization.
        Built for drug discovery, regulatory review, medical affairs, and commercial intelligence teams.
    </div>
    <div class="pill-row">
        <div class="pill">Semantic Search</div>
        <div class="pill">Vector Retrieval</div>
        <div class="pill">AI Services</div>
        <div class="pill">Pharma Catalog Insights</div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Embedding Model</div>
        <div class="metric-value">NV EmbedQA</div>
        <div class="metric-sub">{EMBEDDING_MODEL}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Chat Model</div>
        <div class="metric-value">Mistral 7B</div>
        <div class="metric-sub">{CHAT_MODEL}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Catalog Scope</div>
        <div class="metric-value">{CB_BUCKET}</div>
        <div class="metric-sub">{CB_SCOPE}.{CB_COLLECTION}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Vector Config</div>
        <div class="metric-value">Top K {TOP_K}</div>
        <div class="metric-sub">{VECTOR_METRIC} · nprobes {NPROBES}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Try prompts like: "Give me Respiratory-class drugs in Phase III with side effects" or "Find approved endocrine drugs with pregnancy category A."</div>',
    unsafe_allow_html=True
)

try:
    cluster = get_cluster()
    embedding_client = get_embedding_client()
    chat_client = get_chat_client()
except Exception as e:
    st.error(
        "Initialization failed while connecting to Couchbase. "
        "This is usually related to WAN latency, DNS, TLS, or cluster bootstrap timing.\n\n"
        f"Details: {e}"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🧑‍⚕️" if msg["role"] == "user" else "🧬"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a grounded clinical catalog question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧬"):
        with st.spinner("Searching catalog and composing grounded pharma answer..."):
            try:
                query_vector = create_query_embedding(embedding_client, prompt)
                rows = retrieve_documents(cluster, query_vector, prompt, TOP_K)
                answer = answer_with_guardrails(chat_client, prompt, rows)

                if rows:
                    with st.expander(f"Retrieved records ({len(rows)} matched)", expanded=False):
                        for row in rows:
                            st.markdown(f"""
                            <div class="record-card">
                                <div class="record-title">{row.get('brand_name', 'Unknown Brand')} · {row.get('doc_id', 'N/A')}</div>
                                <div class="record-meta">
                                    <b>Generic:</b> {row.get('generic_name', 'N/A')}<br>
                                    <b>Class:</b> {row.get('therapeutic_class', 'N/A')}<br>
                                    <b>Status:</b> {row.get('regulatory_status', 'N/A')}<br>
                                    <b>Manufacturer:</b> {row.get('manufacturer', 'N/A')}<br>
                                    <b>Distance:</b> {round(row.get('distance', 0), 4)}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Application error: {e}")