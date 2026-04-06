# app.py
import streamlit as st
import os, io, re, csv, json, glob, random, datetime
import numpy as np
from io import StringIO
from pathlib import Path

import pdfplumber

# LangChain (RAG)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Gemini (chat 전용; 임베딩은 폴백 포함)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# HF 임베딩 폴백
try:
    from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings as HFEmbeddings

# Matching용
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch


# ===== 기본 설정 =====
st.set_page_config(page_title="과제 공고문 요약·추천", layout="wide")
st.title("과제 공고문 요약·매칭기")
st.markdown("---")

# ===== 경로/환경 세팅 =====
REPO_ROOT   = Path(__file__).resolve().parent
PROFILES_DIR = REPO_ROOT / "profiles"
UPLOAD_DIR   = REPO_ROOT / "upload_pdf"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 기본 프로필 파일: repo에 포함(권장)
DEFAULT_PROFILES_PATH = PROFILES_DIR / "profiles_updated.jsonl"

# (선택) 공고 메타 저장소. 없으면 자동 스킵
DEFAULT_BASE_DATA_PATH = REPO_ROOT / "data" / "rfp_archive"

# 사이드바
st.sidebar.header("API Key 설정")
api_key = st.sidebar.text_input("Google AI Studio API 키", type="password")

st.sidebar.header("프로필 파일")
profiles_path_str = st.sidebar.text_input(
    "교수 프로필 JSONL 경로 (repo 상대/절대 모두 가능)",
    value=str(DEFAULT_PROFILES_PATH)
)
profiles_path = Path(profiles_path_str)

st.sidebar.caption("※ PDF 업로드 즉시 요약→매칭이 자동으로 실행됩니다.")

if not api_key:
    st.sidebar.warning("API 키를 입력해주세요. (https://aistudio.google.com/app/apikey)")
    st.stop()

# 환경 변수
os.environ["GOOGLE_API_KEY"] = api_key
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 상수
BASE_DATA_PATH = Path(st.secrets.get("BASE_DATA_PATH", DEFAULT_BASE_DATA_PATH))
SBERT_MODEL_FOR_MATCH = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
HF_EMBED_MODEL_FOR_RAG = SBERT_MODEL_FOR_MATCH
TOP_K_PREVIEW = 30

YEAR_WEIGHTS = {2025: 1.5, 2024: 1.2, 2023: 1.1}
DEFAULT_YEAR_WEIGHT = 1.0
STAGE1_WEIGHTS = np.array([0.7, 0.1, 0.1, 0.1], dtype=float)  # (major, researchs, projects, fingerprints)
STAGE2_WEIGHTS = np.array([0.2, 0.5, 0.2, 0.1], dtype=float)  # (major, research_year, projects, fingerprints)


# ===== 요약 프롬프트 =====
def get_prompt_template():
    return """
## 지시사항 (Instruction)
당신은 국가 연구개발 과제 공고문을 분석하여 핵심 정보를 추출하는 **전문 연구 분석가**입니다.  
주어진 공고문 텍스트를 철저히 검토한 뒤, 아래 제시된 **요약 양식**에 따라 각 항목을 **정확하고 상세하게** 작성해주세요.  
공고문 곳곳에 분산된 정보를 종합하여 내용을 구성하며, 누락이 없도록 주의합니다.  

**가장 중요하게, 작성 시 다음 원칙을 반드시 준수하여 일관되고 올바른 어법과 문법을 유지해주십시오.**
- **명확성 및 간결성:** 불필요한 수식어나 반복적인 표현을 피하고, 핵심 내용을 간결하고 명확하게 전달합니다.
- **전문적이고 객관적인 어조:** 주관적인 판단이나 감정적인 표현 없이, 전문적이고 객관적인 어조를 유지합니다.
- **문체 통일성:** 모든 항목에 걸쳐 통일된 문체와 표현 방식을 사용하여 읽는 이에게 일관된 인상을 줍니다.
- **정확한 용어 사용:** 공고문에 명시된 전문 용어를 정확하게 사용하며, 오타 및 비문이 없도록 합니다.

---

## 입력 데이터 (RFP 원문)
{context}

---

## 과제 (Task)
국가 연구개발 과제 공고문을 분석하여 핵심 정보를 아래 `<요약 양식>`에 따라 구조화된 형태로 정리해주세요.  
각 항목은 반드시 공고문에 기반하여 작성해야 하며, 추정이나 유추는 금지합니다.

(아래는 사용자 질문/추가 지시입니다)
{input}

---

## 출력 양식 (요약 결과)

[추출 결과]

### 과제 목표
- 과제가 달성하고자 하는 최종적이고 핵심적인 목표를 명확하게 기술합니다.
- 필요 시, 정성적 목표와 정량적 목표를 구분하여 서술합니다.

### 연구 기간
- 전체 연구 기간과 시작/종료 연월을 명확히 표기합니다. (예: 2025.07.01 ~ 2029.12.31 (총 54개월))

### 과제 예산
- 총 연구개발비, 정부지원금, 민간부담금(기관부담금) 등 예산 관련 정보를 상세히 기재합니다.

### 지원 자격 및 형태
- **지원 자격:** 기업(대기업, 중견기업, 중소기업), 대학, 연구기관, 협회 등 지원 가능한 주체를 구체적으로 명시합니다.
- **지원 형태:** 주관기관, 공동연구기관, 위탁연구기관 등 참여 형태와 컨소시엄 구성 가능 여부를 명확히 기재합니다.

### 공고 요약
- 과제의 추진 배경, 필요성, 핵심 목표 및 주요 연구 내용 등을 종합하여 5문장 이내로 요약합니다.
- 기술적 특징이나 정책적 의의가 있다면 함께 기술합니다.

### 사업 내용
- 과제를 수행하는 데 요구되는 기술적 핵심 요소와 주요 추진 내용을 상세히 정리합니다.

### 관련 기술/산업 동향
- 본 과제가 속한 기술 분야(예: 소형모듈원자로(SMR), 인공지능(AI), 바이오헬스 등)의 최신 기술 트렌드, 시장 동향, 정책적 중요성 등을 공고문 내용을 기반으로 1~2문장 이내로 요약합니다.

### 기대 효과 및 활용 방안
- **기술적 기대효과:** 연구 성공 시 확보할 수 있는 기술 수준 및 파급 효과
- **경제적/산업적 기대효과:** 수출, 매출, 고용, 시장 창출 등 경제·산업적 기여 방안 및 기대 효과
- **활용 방안:** 개발된 기술이 실제 어디에 어떻게 적용될 수 있는지를 구체적으로 서술합니다.

### 주요 평가 항목/중점 사항
- 선정 평가 기준, 우대사항, 가점 항목 등 공고문에 명시된 핵심 평가 요소를 리스트 형태로 정리합니다.

### 키워드
- 과제의 핵심 기술, 목표, 적용 분야 등을 나타내는 핵심 단어를 10개 이하로 작성하고 쉼표(,)로 구분합니다.

※ 위 항목까지 작성한 후, 동일한 내용을 반복하거나 덧붙이지 마세요. 출력은 여기서 끝입니다.
"""


# ===== 유틸 =====
def clean_final_output(raw_text: str) -> str:
    start_marker = "### 과제 목표"
    i = raw_text.find(start_marker)
    if i == -1:
        j = raw_text.find("###")
        return raw_text.strip() if j == -1 else raw_text[j:].strip()
    return raw_text[i:].strip()


def find_file_and_get_info(base_path: Path, uploaded_filename: str):
    """
    base_path 아래에서 uploaded_filename과 동일한 파일을 찾아,
    같은 폴더의 department_name.txt / notice_link.txt 내용을 읽어온다.
    base_path가 존재하지 않으면 '정보 없음' 반환.
    """
    base_path = Path(base_path)
    if not base_path.exists():
        return False, "정보 없음", "정보 없음", "정보 없음"

    for file in base_path.rglob("*"):
        if file.is_file() and file.name == uploaded_filename:
            file_folder = file.parent
            folder_name = file_folder.name
            dep_path = file_folder / "department_name.txt"
            link_path = file_folder / "notice_link.txt"
            try:
                department = dep_path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                department = "정보 없음"
            try:
                link = link_path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                link = "정보 없음"
            return True, department, link, folder_name
    return False, "정보 없음", "정보 없음", "정보 없음"


def extract_text_from_pdf(pdf_bytes):
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        if not text.strip():
            raise ValueError("No text found in PDF.")
        return [Document(page_content=text)]
    except Exception as e:
        st.error(f"PDF 텍스트 추출 중 오류 발생: {e}")
        return None


def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def csv_bytes_from_rows(rows, fieldnames):
    sio = StringIO()
    writer = csv.DictWriter(sio, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return sio.getvalue().encode("utf-8-sig")


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        out = []
        for v in val.values():
            out += v if isinstance(v, list) else [str(v)]
        return out
    return [] if val is None else [str(val)]


def load_profiles(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"프로필 파일을 찾을 수 없습니다: {path}")
    profiles = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            p['researchs']    = normalize_list(p.get('researchs', []))
            p['projects']     = normalize_list(p.get('projects', []))
            p['fingerprints'] = normalize_list(p.get('fingerprints', []))
            p['email']        = p.get('email', "")
            profiles.append(p)
    return profiles


def find_elbow_threshold(scores: np.ndarray) -> float:
    """2차 차분 기반 엘보우 지점 탐지"""
    if scores.size == 0:
        return 0.0
    if scores.size == 1:
        return float(scores[0])
    s = np.sort(scores)[::-1]
    sd2 = np.diff(s, n=2)
    idx = int(np.argmax(np.abs(sd2))) + 1
    return float(s[idx])


def run_matching(summary_text: str, profiles_file: str, top_k_preview: int = TOP_K_PREVIEW):
    """
    summary_text 기반으로 교수 프로필 매칭.
    화면에는 상위 후보들의 score와 label(True/False)을 같이 보여주고,
    CSV는 여전히 label=True(추천)만 저장.
    """
    set_seeds(42)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    profiles = load_profiles(profiles_file)

    # SBERT 로드 & 문서(=요약) 임베딩
    sbert = SentenceTransformer(SBERT_MODEL_FOR_MATCH, device=DEVICE)
    emb_doc = sbert.encode([summary_text], convert_to_numpy=True, normalize_embeddings=True)

    # ----- Stage1 (major 비중 높음) -----
    majors = [p.get('major','') for p in profiles]
    emb_majors = sbert.encode(majors, convert_to_numpy=True, normalize_embeddings=True)
    sim_major_all = cosine_similarity(emb_doc, emb_majors)[0]

    texts_res_all = [' | '.join(p['researchs']) for p in profiles]
    emb_res_all = sbert.encode(texts_res_all, convert_to_numpy=True, normalize_embeddings=True)
    sim_res_all = cosine_similarity(emb_doc, emb_res_all)[0]

    texts_proj_all = [' | '.join(p['projects']) for p in profiles]
    emb_proj_all = sbert.encode(texts_proj_all, convert_to_numpy=True, normalize_embeddings=True)
    sim_proj_all = cosine_similarity(emb_doc, emb_proj_all)[0]

    texts_fp_all = [' | '.join(p['fingerprints']) for p in profiles]
    emb_fp_all = sbert.encode(texts_fp_all, convert_to_numpy=True, normalize_embeddings=True)
    sim_fp_all = cosine_similarity(emb_doc, emb_fp_all)[0]

    stage1_scores = (
        STAGE1_WEIGHTS[0] * sim_major_all +
        STAGE1_WEIGHTS[1] * sim_res_all  +
        STAGE1_WEIGHTS[2] * sim_proj_all +
        STAGE1_WEIGHTS[3] * sim_fp_all
    )

    k150 = min(150, len(profiles))
    idxs1 = np.argsort(stage1_scores)[::-1][:k150]
    cand1 = [profiles[i] for i in idxs1]

    # cand1에 해당하는 유사도 subset
    sim_major = sim_major_all[idxs1]
    sim_proj  = sim_proj_all[idxs1]
    sim_fp    = sim_fp_all[idxs1]

    # ----- Stage2 (연도 가중 researchs) -----
    current_year = datetime.datetime.now().year
    years = list(range(current_year, current_year - 25, -1))

    raw_year_weights = np.array(
        [YEAR_WEIGHTS.get(y, DEFAULT_YEAR_WEIGHT) for y in years],
        dtype=float
    )
    year_weights = raw_year_weights / raw_year_weights.sum()

    all_year_sims = []
    for p in cand1:
        sims_per_year = []
        for y in years:
            entries = [e for e in p['researchs'] if str(y) in e]
            text = ' | '.join(entries) if entries else ''
            if text:
                sim_val = cosine_similarity(
                    emb_doc,
                    sbert.encode([text], convert_to_numpy=True, normalize_embeddings=True)
                )[0][0]
            else:
                sim_val = 0.0
            sims_per_year.append(sim_val)
        all_year_sims.append(sims_per_year)
    all_year_sims = np.array(all_year_sims)

    research_year_scores = np.dot(all_year_sims, year_weights)

    # Stage2 최종 스코어
    stage2_scores = (
        STAGE2_WEIGHTS[0] * sim_major +
        STAGE2_WEIGHTS[1] * research_year_scores +
        STAGE2_WEIGHTS[2] * sim_proj +
        STAGE2_WEIGHTS[3] * sim_fp
    )

    # ----- 엘보우 컷 -----
    thr = find_elbow_threshold(stage2_scores)
    labels = stage2_scores >= thr

    # 점수 기준 정렬 (내림차순)
    order = np.argsort(stage2_scores)[::-1]

    # 전체 후보(컷 전) + 추천 후보(컷 후)
    rows_all = []
    rows_rec = []
    for rank, idx in enumerate(order, start=1):
        p = cand1[idx]
        row = {
            "rank":  rank,
            "name":  p.get("name", ""),
            "major": p.get("major", ""),
            "email": p.get("email", ""),
            "score": float(stage2_scores[idx]),
            "label": bool(labels[idx]),
        }
        rows_all.append(row)
        if labels[idx]:
            rows_rec.append(row)

    # 화면용: 상위 N명 (score + label 둘 다 보여줌)
    preview_all = rows_all[:top_k_preview]

    # CSV는 여전히 label=True만
    fieldnames = ["rank", "name", "major", "email", "score", "label"]
    csv_rec = csv_bytes_from_rows(rows_rec, fieldnames)

    meta = {
        "threshold": float(thr),
        "n_candidates": len(cand1),
        "n_recommended": len(rows_rec),
    }

    return preview_all, csv_rec, meta, rows_rec


# ===== RAG 요약 + 매칭: 업로드 즉시 자동 실행 =====
uploaded_file = st.file_uploader("요약·매칭할 PDF 공고문 파일을 업로드하세요.", type="pdf")

if uploaded_file:
    # ── 업로드 파일을 repo/upload_pdf/ 에 저장 ──
    safe_name = uploaded_file.name.replace("/", "_").replace("\\", "_")
    # 충돌 방지용 타임스탬프(원하면 지워도 됨)
    stem, suf = os.path.splitext(safe_name)
    unique_name = f"{stem}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{suf}"
    pdf_path = UPLOAD_DIR / unique_name
    pdf_path.write_bytes(uploaded_file.getvalue())

    st.success(f"📁 업로드 파일 저장: {pdf_path.relative_to(REPO_ROOT)}")

    # (선택) BASE_DATA_PATH가 있을 때만 메타 조회 시도
    if BASE_DATA_PATH.exists():
        found, department, notice_link, folder_title = find_file_and_get_info(BASE_DATA_PATH, uploaded_file.name)
        if found:
            st.info(
                f"**📂 공고 제목:** {folder_title}  \n"
                f"**🏢 주관 기관:** {department}  \n"
                f"**🔗 공고 링크:** {notice_link}"
            )

    # ① PDF → 텍스트 (저장한 파일에서 다시 로드)
    with st.spinner("① PDF 분석 중 (텍스트 추출)…"):
        documents = extract_text_from_pdf(pdf_path.read_bytes())
        if not documents:
            st.stop()

    # ② RAG 준비
    with st.spinner("② RAG 준비 중 (청크/임베딩/인덱싱)…"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        texts = splitter.split_documents(documents)
        if not texts:
            st.error("텍스트를 처리 가능한 단위로 분할할 수 없습니다.")
            st.stop()

        # Google 임베딩 우선, 실패 시 HF 폴백
        try:
            g_emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_documents(texts, g_emb)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                st.warning("Google 임베딩 쿼터 초과로 로컬 임베딩으로 전환합니다.")
                hf_emb = HFEmbeddings(
                    model_name=HF_EMBED_MODEL_FOR_RAG,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                vector_store = FAISS.from_documents(texts, hf_emb)
            else:
                raise

        retriever = vector_store.as_retriever(search_kwargs={"k": 7})

    # ③ Gemini 요약
    with st.spinner("③ 요약 생성 중 (Gemini)…"):
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0)
        stuff_prompt = ChatPromptTemplate.from_template(get_prompt_template())
        chain = (
            {"context": retriever | RunnableLambda(_format_docs), "input": RunnablePassthrough()}
            | stuff_prompt
            | llm
            | StrOutputParser()
        )
        question = "이 공고문의 내용을 프롬프트의 '출력 양식'에 맞춰서 아주 상세하게 요약해줘."
        result_text = chain.invoke(question)
        summary = clean_final_output(result_text)

    st.success("✅ 요약 완료")
    st.text_area("📌 최종 요약 결과", summary, height=450)

    # ④ SBERT 매칭
    if not profiles_path.exists():
        st.error(f"프로필 파일을 찾을 수 없습니다: {profiles_path}")
        st.stop()

    with st.spinner("④ 요약 기반 교수 매칭 계산 중 (SBERT)…"):
        try:
            preview_rec, csv_rec, meta, recommended_list = run_matching(
                summary_text=summary,
                profiles_file=str(profiles_path),  # 내부에서 open(str) 사용
                top_k_preview=TOP_K_PREVIEW
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success(
        f"✅ 매칭 완료 | 임계값={meta['threshold']:.4f} | "
        f"후보 {meta['n_candidates']}명 중 추천 {meta['n_recommended']}명"
    )

    # 최종 추천 테이블 표시 (상위 미리보기)
    st.subheader("추천 대상 미리보기 (상위 후보)")
    st.dataframe(preview_rec, use_container_width=True)

    # 추천 CSV 다운로드 (label=True만 저장됨)
    st.download_button(
        "추천 후보 CSV 다운로드",
        data=csv_rec,
        file_name=f"match_results_recommended_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.info("👆 PDF를 업로드하면 자동으로 **upload_pdf/**에 저장된 뒤 요약 → 매칭이 실행됩니다.")
