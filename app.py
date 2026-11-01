import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from google.oauth2 import service_account
from langchain_google_vertexai import ChatVertexAI
from langchain_community.retrievers import BM25Retriever

st.set_page_config(page_title="AI-Biz 파트너", layout="wide")
st.title("🤖 기업 맞춤형 정부지원사업 AI 추천기")

# 2. 문서 처리 및 AI 셋업
@st.cache_resource
def setup_ai():
    # 2-1. 폴더에서 PDF 읽기
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()
    
    # 2-2. 문서 쪼개기
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # 2-3. 에러를 원천 차단한 가볍고 강력한 BM25 검색엔진 적용
    retriever = BM25Retriever.from_documents(splits)
    retriever.k = 3

    # 🔥 수정 1: 바깥에서 쓸 수 있도록 결과물을 반환(return)해 줍니다.
    return retriever, len(docs)
    
key_dict = st.secrets["gcp_service_account"]
credentials = service_account.Credentials.from_service_account_info(key_dict)

llm = ChatVertexAI(
    model_name="gemini-1.5-flash", 
    project=key_dict["project_id"],
    location="us-central1",  # 이 줄이 반드시 포함되어야 합니다.
    credentials=credentials,
    temperature=0.1
)

# 3. 공고문 데이터베이스 로딩
st.subheader("📂 공고문 데이터베이스 로딩 중...")
try:
    # 🔥 수정 2: 바깥에서 이미 만든 llm은 빼고, 검색엔진과 문서 개수만 받아옵니다.
    retriever, doc_count = setup_ai()
    st.success(f"성공적으로 총 {doc_count}페이지 분량의 공고문을 학습했습니다! 이제 AI에게 물어보세요.")
except Exception as e:
    st.error(f"오류가 발생했습니다. (에러내용: {e})")
    st.stop()

st.divider()

# 4. 사용자 입력 및 검색 (깔끔하게 개선된 선택형 UI)
st.subheader("🔍 우리 기업에 맞는 사업 찾기")

col1, col2 = st.columns(2)
with col1:
    location = st.radio("🏢 기업 소재지", ["평택시 관내", "경기도 타 시군", "기타 지역"])
with col2:
    industry = st.selectbox("⚙️ 주력 산업 분야", ["반도체/디스플레이", "제약/바이오", "인공지능(AI)/소프트웨어", "일반 기계/제조업", "기타"])

support_type = st.multiselect("💡 희망 지원 분야 (복수 선택 가능)", 
                              ["해외 플랫폼 입점", "국내외 전시회 참가", "우수기업 선정/인증", "시제품 제작", "R&D 기술개발"])

user_input = ""
if support_type:
    support_str = ", ".join(support_type)
    user_input = f"{location}에 위치한 {industry} 분야 기업입니다. {support_str} 관련 지원사업을 찾고 있습니다."
    st.info(f"🤖 AI 분석 조건: '{user_input}'")

if st.button("추천 사업 찾기"):
    if user_input:
        with st.spinner("AI가 공고문을 분석하여 맞춤형 사업을 찾고 있습니다..."):
            # 이제 retriever가 정상적으로 준비되어 에러 없이 문서를 찾습니다.
            relevant_docs = retriever.invoke(user_input)
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            prompt = f"""
            당신은 친절한 기업 지원사업 추천 전문가입니다.
            아래 [참고자료]는 현재 우리 기관이 지원 중인 사업들의 공고문 내용입니다.
            이 자료만을 바탕으로 사용자의 상황에 가장 적합한 지원사업을 찾아 추천해주세요.
            
            [참고자료]
            {context_text}
            
            [사용자 상황]
            {user_input}
            """
            
            result = llm.invoke(prompt)
            st.success("분석 완료!")
            st.markdown("### 💡 AI의 추천 결과")
            
            # 1. AI 답변이 복잡한 괄호 구조(리스트)로 넘어오면 순수 텍스트만 추출
            if isinstance(result.content, list):
                final_answer = result.content[0].get("text", "")
            else:
                final_answer = result.content
                
            # 2. 추출한 텍스트를 깔끔한 문서(마크다운) 형식으로 화면에 출력
            st.markdown(final_answer)
