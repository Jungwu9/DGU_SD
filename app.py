

import os
import io
import re
import csv
import json
import glob
import datetime
from pathlib import Path
from io import StringIO
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
import pdfplumber
import torch
from sentence_transformers import SentenceTransformer

# LangChain (RAG)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# RAG 임베딩은 기존 코드와 동일하게 HF 기반만 사용
try:
    from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings as HFEmbeddings


# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="과제 공고문 요약·추천", layout="wide")
st.title("과제 공고문 요약·매칭기")
st.markdown("---")

REPO_ROOT = Path(__file__).resolve().parent
PROFILES_DIR = REPO_ROOT / "profiles"
EMB_DIR = REPO_ROOT / "data" / "emb"
UPLOAD_DIR = REPO_ROOT / "upload_pdf"
DEFAULT_BASE_DATA_PATH = REPO_ROOT / "data" / "rfp_archive"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_API_KEY = ""
DEFAULT_PROFILES_PATH = PROFILES_DIR / "profiles_updated_exp2.jsonl"
DEFAULT_EMB_DIR = EMB_DIR
DEFAULT_HF_EMBED_MODEL_FOR_RAG = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SBERT_MODEL_FOR_MATCH = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TOP_K_PREVIEW = 30
RETRIEVER_K = 7


def _safe_secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# =========================
# 경로 유틸
# =========================
def resolve_path(path_str: str, default_path: Path) -> Path:
    """
    - 빈 값이면 default_path
    - 절대경로면 그대로 사용
    - 상대경로면 repo root 기준으로 해석
    """
    text = (path_str or "").strip()
    if not text:
        return default_path

    p = Path(text).expanduser()
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


# =========================
# 사이드바
# =========================
st.sidebar.header("API Key 설정")
default_api_key = _safe_secret("GOOGLE_API_KEY", DEFAULT_API_KEY)
api_key = st.sidebar.text_input("Google AI Studio API 키", type="password", value=default_api_key)

st.sidebar.header("프로필/임베딩 경로")
profiles_path_input = st.sidebar.text_input(
    "교수 프로필 JSONL 경로 (repo 상대/절대 모두 가능)",
    value=str(DEFAULT_PROFILES_PATH),
)
emb_dir_input = st.sidebar.text_input(
    "사전 임베딩 폴더 경로 (repo 상대/절대 모두 가능)",
    value=str(DEFAULT_EMB_DIR),
)

profiles_path = resolve_path(profiles_path_input, DEFAULT_PROFILES_PATH)
emb_dir = resolve_path(emb_dir_input, DEFAULT_EMB_DIR)
base_data_path = resolve_path(
    _safe_secret("BASE_DATA_PATH", str(DEFAULT_BASE_DATA_PATH)),
    DEFAULT_BASE_DATA_PATH,
)


# =========================
# 자동 파일 선택
# =========================
def pick_precomputed_files(emb_dir_path: Path) -> Tuple[Path, Path]:
    npz_list = sorted(glob.glob(str(emb_dir_path / "profiles_embeds_*.npz")))
    json_list = sorted(glob.glob(str(emb_dir_path / "profiles_meta_*.json")))

    if not npz_list or not json_list:
        return Path(""), Path("")

    tag2json = {}
    for p in json_list:
        m = re.search(r"profiles_meta_(.+?)\.json$", p)
        if m:
            tag2json[m.group(1)] = p

    for npz in npz_list:
        m = re.search(r"profiles_embeds_(.+?)\.npz$", npz)
        if not m:
            continue
        tag = m.group(1)
        if tag in tag2json:
            return Path(npz), Path(tag2json[tag])

    return Path(npz_list[0]), Path(json_list[0])


picked_npz_path, picked_meta_path = pick_precomputed_files(emb_dir)
emb_npz_input = st.sidebar.text_input(
    "사전 임베딩(.npz) 경로 (repo 상대/절대 모두 가능)",
    value=str(picked_npz_path) if str(picked_npz_path) else "",
)
meta_json_input = st.sidebar.text_input(
    "메타(.json) 경로 (repo 상대/절대 모두 가능)",
    value=str(picked_meta_path) if str(picked_meta_path) else "",
)

emb_npz_path = resolve_path(emb_npz_input, picked_npz_path) if emb_npz_input else picked_npz_path
meta_json_path = resolve_path(meta_json_input, picked_meta_path) if meta_json_input else picked_meta_path

st.sidebar.caption("※ PDF 업로드 시 요약 → 매칭이 자동 실행됩니다.")

if not api_key:
    st.sidebar.warning("API 키를 입력해주세요. (또는 secrets에 GOOGLE_API_KEY 설정)")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# =========================
# 프롬프트
# =========================
def get_prompt_template() -> str:
    return """
## 지시사항 (Instruction)
당신은 국가 연구개발 과제 공고문을 분석하여 핵심 정보를 추출하는 전문 연구 분석가입니다.
주어진 공고문 텍스트를 철저히 검토한 뒤, 아래 제시된 요약 양식에 따라 각 항목을 정확하고 상세하게 작성해주세요.
공고문에 없는 내용은 추정하지 말고, 반드시 공고문 내용에 기반하여 정리하세요.

---

## 입력 데이터 (RFP 원문)
{context}

---

## 과제 (Task)
국가 연구개발 과제 공고문을 분석하여 핵심 정보를 아래 출력 양식에 따라 구조화해 주세요.
(아래는 사용자 질문/추가 지시입니다)
{input}

---

## 출력 양식 (요약 결과)

[추출 결과]

### 과제 목표
- ...

### 연구 기간
- ...

### 과제 예산
- ...

### 지원 자격 및 형태
- ...

### 공고 요약
- ...

### 사업 내용
- ...

### 관련 기술/산업 동향
- ...

### 기대 효과 및 활용 방안
- ...

### 주요 평가 항목/중점 사항
- ...

### 키워드
- ...

※ 위 항목까지만 작성하고 반복 설명은 하지 마세요.
"""


# =========================
# 공통 유틸
# =========================
def clean_final_output(raw_text: str) -> str:
    start_marker = "### 과제 목표"
    i = raw_text.find(start_marker)
    if i == -1:
        j = raw_text.find("###")
        return raw_text.strip() if j == -1 else raw_text[j:].strip()
    return raw_text[i:].strip()


def extract_text_from_pdf(pdf_bytes: bytes) -> Optional[List[Document]]:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        if not text.strip():
            raise ValueError("No text found in PDF.")

        return [Document(page_content=text)]
    except Exception as e:
        st.error(f"PDF 텍스트 추출 중 오류 발생: {e}")
        return None


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def csv_bytes_from_rows(rows: List[Dict], fieldnames: List[str]) -> bytes:
    sio = StringIO()
    writer = csv.DictWriter(sio, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return sio.getvalue().encode("utf-8-sig")


def find_file_and_get_info(base_path: Path, uploaded_filename: str):
    """
    base_path 아래에서 uploaded_filename과 동일한 파일을 찾고,
    같은 폴더의 department_name.txt / notice_link.txt 내용을 읽어온다.
    없으면 정보 없음 반환.
    """
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


# =========================
# SBERT / 사전 임베딩 캐시
# =========================
@st.cache_resource(show_spinner=False)
def load_sbert_cached(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    _ = model.encode(["warmup"], normalize_embeddings=True)
    return model, device


@st.cache_data(show_spinner=False)
def load_precomputed(npz_path_str: str, meta_path_str: str):
    npz_path = Path(npz_path_str)
    meta_path = Path(meta_path_str)

    if not (npz_path.exists() and meta_path.exists()):
        raise FileNotFoundError("사전 임베딩(.npz) 또는 메타(.json) 경로가 올바르지 않습니다.")

    data = np.load(npz_path)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    for key in ["E_maj", "E_res", "E_prj", "E_fp"]:
        if key not in data:
            raise KeyError(f"npz 파일에 '{key}' 키가 없습니다.")

    if "rows" not in meta:
        raise KeyError("meta json 파일에 'rows' 키가 없습니다.")

    return data["E_maj"], data["E_res"], data["E_prj"], data["E_fp"], meta["rows"]


# =========================
# 매칭 로직 (첨부 코드 기준)
# =========================
def run_matching_precomputed(
    summary_text: str,
    model_name: str,
    npz_path_str: str,
    meta_path_str: str,
    top_k: int = TOP_K_PREVIEW,
):
    sbert, _ = load_sbert_cached(model_name)
    E_maj, E_res, E_prj, E_fp, rows = load_precomputed(npz_path_str, meta_path_str)

    q = sbert.encode([summary_text], normalize_embeddings=True)[0]  # shape: (d,)

    sim_major = E_maj @ q
    sim_res = E_res @ q
    sim_prj = E_prj @ q
    sim_fp = E_fp @ q

    score = 0.7 * sim_major + 0.1 * sim_res + 0.1 * sim_prj + 0.1 * sim_fp
    idx = np.argsort(score)[::-1][: min(top_k, len(rows))]

    results = []
    for rank, i in enumerate(idx, 1):
        info = rows[i] if i < len(rows) else {}
        results.append(
            {
                "rank": rank,
                "name": info.get("name", ""),
                "major": info.get("major", ""),
                "email": info.get("email", ""),
                "score": float(score[i]),
            }
        )

    fieldnames = ["rank", "name", "major", "email", "score"]
    csv_data = csv_bytes_from_rows(results, fieldnames)

    return results, csv_data


# =========================
# (첨부 코드 기준) Top 교수 상세 정보
# =========================
@st.cache_data(show_spinner=False)
def load_profiles_index(jsonl_path_str: str):
    by_email, by_name = {}, {}
    path = Path(jsonl_path_str)
    if not path.exists():
        return by_email, by_name

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            email = str(obj.get("email") or obj.get("Email") or obj.get("e-mail") or obj.get("mail") or "").strip()
            name = str(obj.get("name") or obj.get("Name") or obj.get("prof_name") or obj.get("professor") or "").strip()

            if email:
                by_email.setdefault(email.lower(), obj)
            if name:
                by_name.setdefault(name, obj)

    return by_email, by_name


def extract_paper_titles(profile: dict, top_n: int = 5) -> List[str]:
    if not profile or not isinstance(profile, dict):
        return []

    candidate_keys = [
        "papers", "paper_list", "publications", "publication_list",
        "scopus_papers", "scopus_publications", "pubs",
        "논문", "논문목록", "논문리스트", "publication", "Publications",
    ]

    papers = None

    for k in candidate_keys:
        v = profile.get(k)
        if isinstance(v, list) and v:
            papers = v
            break

    if papers is None:
        for nk in ["scopus", "Scopus", "profile", "Profile", "data", "Data"]:
            v = profile.get(nk)
            if isinstance(v, dict):
                for kk in candidate_keys:
                    vv = v.get(kk)
                    if isinstance(vv, list) and vv:
                        papers = vv
                        break
            if papers is not None:
                break

    if papers is None:
        return []

    title_keys = ["title", "paper_title", "document_title", "dc:title", "논문명", "제목", "Title"]

    def _get_title(item):
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for tk in title_keys:
                if tk in item and item[tk]:
                    return str(item[tk]).strip()
        return ""

    def _get_citations(item):
        if not isinstance(item, dict):
            return None
        for ck in ["citedby_count", "citation_count", "citations", "citedby", "cited_by", "CitedByCount"]:
            if ck in item and item[ck] is not None:
                try:
                    return float(item[ck])
                except Exception:
                    pass
        return None

    def _get_year(item):
        if not isinstance(item, dict):
            return None
        for yk in ["year", "pub_year", "publication_year", "coverDate", "date", "issued", "Year"]:
            if yk in item and item[yk]:
                val = item[yk]
                if isinstance(val, str):
                    m = re.search(r"(19|20)\d{2}", val)
                    if m:
                        return int(m.group(0))
                try:
                    return int(val)
                except Exception:
                    pass
        return None

    normalized = []
    for p in papers:
        title = _get_title(p)
        if title:
            normalized.append({"title": title, "cit": _get_citations(p), "year": _get_year(p)})

    if not normalized:
        return []

    normalized.sort(
        key=lambda x: (
            -(x["cit"] if x["cit"] is not None else -1),
            -(x["year"] if x["year"] is not None else -1),
        ),
        reverse=False,
    )
    normalized = sorted(
        normalized,
        key=lambda x: (
            x["cit"] is None,
            -(x["cit"] if x["cit"] is not None else -1),
            x["year"] is None,
            -(x["year"] if x["year"] is not None else -1),
        ),
    )

    return [x["title"] for x in normalized[:top_n]]


def explain_top_professor_components(
    summary_text: str,
    model_name: str,
    npz_path_str: str,
    meta_path_str: str,
    prof_email: str = "",
    prof_name: str = "",
):
    try:
        sbert, _ = load_sbert_cached(model_name)
        E_maj, E_res, E_prj, E_fp, rows = load_precomputed(npz_path_str, meta_path_str)
        q = sbert.encode([summary_text], normalize_embeddings=True)[0]

        idx = None
        email_norm = (prof_email or "").strip().lower()
        name_norm = (prof_name or "").strip()

        if email_norm:
            for i, row in enumerate(rows):
                em = str(row.get("email", "")).strip().lower()
                if em and em == email_norm:
                    idx = i
                    break

        if idx is None and name_norm:
            for i, row in enumerate(rows):
                nm = str(row.get("name", "")).strip()
                if nm and nm == name_norm:
                    idx = i
                    break

        if idx is None:
            return None

        sim_major = float(E_maj[idx] @ q)
        sim_res = float(E_res[idx] @ q)
        sim_prj = float(E_prj[idx] @ q)
        sim_fp = float(E_fp[idx] @ q)

        contrib = {
            "major": 0.7 * sim_major,
            "research": 0.1 * sim_res,
            "project": 0.1 * sim_prj,
            "fingerprint": 0.1 * sim_fp,
        }
        best_key = max(contrib, key=contrib.get)

        return {
            "sim_major": sim_major,
            "sim_res": sim_res,
            "sim_prj": sim_prj,
            "sim_fp": sim_fp,
            "best": best_key,
        }
    except Exception:
        return None


# =========================
# RAG 벡터스토어 생성 (기존 코드와 동일한 HF 임베딩 사용)
# =========================
def build_vector_store(texts: List[Document]):
    hf_emb = HFEmbeddings(
        model_name=DEFAULT_HF_EMBED_MODEL_FOR_RAG,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(texts, hf_emb)


# =========================
# 메인 UI
# =========================
uploaded_file = st.file_uploader("요약·매칭할 PDF 공고문 파일을 업로드하세요.", type="pdf")

if uploaded_file:
    safe_name = uploaded_file.name.replace("/", "_").replace("\\", "_")
    stem, suffix = os.path.splitext(safe_name)
    unique_name = f"{stem}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    saved_pdf_path = UPLOAD_DIR / unique_name
    saved_pdf_path.write_bytes(uploaded_file.getvalue())

    try:
        relative_saved_path = saved_pdf_path.relative_to(REPO_ROOT)
        st.success(f"📁 업로드 파일 저장: {relative_saved_path}")
    except Exception:
        st.success(f"📁 업로드 파일 저장: {saved_pdf_path}")

    if base_data_path.exists():
        found, department, notice_link, folder_title = find_file_and_get_info(base_data_path, uploaded_file.name)
        if found:
            st.info(
                f"**📂 공고 제목:** {folder_title}  \n"
                f"**🏢 주관 기관:** {department}  \n"
                f"**🔗 공고 링크:** {notice_link}"
            )

    with st.spinner("① PDF 분석 중 (텍스트 추출)…"):
        docs = extract_text_from_pdf(saved_pdf_path.read_bytes())
        if not docs:
            st.stop()

    with st.spinner("② RAG 준비 중 (청크/임베딩/인덱싱)…"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        texts = splitter.split_documents(docs)
        if not texts:
            st.error("텍스트를 처리 가능한 단위로 분할할 수 없습니다.")
            st.stop()

        try:
            vector_store = build_vector_store(texts)
        except Exception as e:
            st.exception(e)
            st.stop()

        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

    with st.spinner("③ 요약 생성 중 (Gemini)…"):
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                generation_config={
                    "seed": 42,
                    "top_p": 1.0,
                    "top_k": 1,
                },
            )

            prompt = ChatPromptTemplate.from_template(get_prompt_template())
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "input": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            question = "이 공고문의 내용을 '출력 양식'에 맞춰 정확하고 간결하게 요약해줘."
            summary_raw = chain.invoke(question)
            summary = clean_final_output(summary_raw)
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success("✅ 요약 완료")
    st.text_area("📌 최종 요약 결과", summary, height=420)

    if not profiles_path.exists():
        st.error(f"프로필 파일을 찾을 수 없습니다: {profiles_path}")
        st.stop()

    if not emb_npz_path.exists() or not meta_json_path.exists():
        st.error(
            "사전 임베딩(.npz) 또는 메타(.json) 파일을 찾을 수 없습니다.\n"
            f"- npz: {emb_npz_path}\n"
            f"- json: {meta_json_path}"
        )
        st.stop()

    with st.spinner("④ 요약 기반 교수 매칭 계산 중… (사전 임베딩 사용)"):
        try:
            results, csv_data = run_matching_precomputed(
                summary_text=summary,
                model_name=SBERT_MODEL_FOR_MATCH,
                npz_path_str=str(emb_npz_path),
                meta_path_str=str(meta_json_path),
                top_k=TOP_K_PREVIEW,
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success("✅ 매칭 완료")
    st.subheader("추천 대상 미리보기 (상위)")
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "추천 결과 CSV 다운로드",
        data=csv_data,
        file_name=f"match_results_top{len(results)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # 1순위 교수 상세 정보 표시
    if results:
        top_prof = results[0]
        by_email, by_name = load_profiles_index(str(profiles_path))

        profile = None
        top_email = str(top_prof.get("email", "")).strip().lower()
        top_name = str(top_prof.get("name", "")).strip()

        if top_email and top_email in by_email:
            profile = by_email[top_email]
        elif top_name and top_name in by_name:
            profile = by_name[top_name]

        explain = explain_top_professor_components(
            summary_text=summary,
            model_name=SBERT_MODEL_FOR_MATCH,
            npz_path_str=str(emb_npz_path),
            meta_path_str=str(meta_json_path),
            prof_email=top_prof.get("email", ""),
            prof_name=top_prof.get("name", ""),
        )

        st.markdown("---")
        st.subheader("1순위 추천 교수 상세")
        st.markdown(f"**이름:** {top_prof.get('name', '')}")
        st.markdown(f"**전공:** {top_prof.get('major', '')}")
        st.markdown(f"**이메일:** {top_prof.get('email', '')}")
        st.markdown(f"**종합 점수:** {top_prof.get('score', 0.0):.4f}")

        if explain:
            st.markdown("**추천 근거 (구성요소 유사도):**")
            st.write(
                {
                    "major": round(explain["sim_major"], 4),
                    "research": round(explain["sim_res"], 4),
                    "project": round(explain["sim_prj"], 4),
                    "fingerprint": round(explain["sim_fp"], 4),
                    "가장 크게 기여한 요소": explain["best"],
                }
            )

        paper_titles = extract_paper_titles(profile, top_n=5) if profile else []
        if paper_titles:
            st.markdown("**대표 논문 Top 5:**")
            for i, title in enumerate(paper_titles, start=1):
                st.markdown(f"{i}. {title}")
        else:
            st.caption("프로필 JSONL에서 대표 논문 정보를 찾지 못했습니다.")

else:
    st.info("👆 PDF를 업로드하면 자동으로 upload_pdf/ 에 저장된 뒤 요약 → 매칭이 실행됩니다.")
