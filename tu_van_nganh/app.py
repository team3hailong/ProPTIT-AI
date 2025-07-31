import streamlit as st
import asyncio

# Đảm bảo event loop tồn tại cho async client
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from utils import load_vectorstore
from dotenv import load_dotenv

load_dotenv()
# --- Sidebar ---
st.title("🎓 Tư vấn ngành học RAG + LangChain + Streamlit")
model_name = st.sidebar.selectbox("Chọn model LLM:", ["models/gemini-2.5-pro"])  # Chỉ dùng Gemini
temp = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.3)

# --- Load or build vectorstore ---
with st.spinner("Đang tải embeddings... 🎯"):
    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 4})

# Initialize chain với Gemini
llm = ChatGoogleGenerativeAI(model=model_name, temperature=temp)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# --- User Input ---

st.header("Thông tin bạn")
user_pref = st.text_area(
    "Mô tả sở thích, kỹ năng, mục tiêu của bạn:",
    height=180,
    placeholder=(
        "Ví dụ:\n"
        "- Sở thích: thích công nghệ, thích làm việc nhóm, thích sáng tạo\n"
        "- Kỹ năng: lập trình Python, giao tiếp tốt, tư duy logic\n"
        "- Mục tiêu: muốn làm việc trong lĩnh vực AI hoặc CNTT\n"
        "- Môn học yêu thích: Toán, Tin học\n"
        "- Điểm mạnh/yếu: mạnh về phân tích, yếu về ghi nhớ lý thuyết\n"
        "Bạn càng cung cấp nhiều thông tin, kết quả tư vấn càng chính xác!"
    )
)

if st.button("Tư vấn ngành học"):
    if not user_pref.strip():
        st.error("Vui lòng nhập thông tin trước khi tư vấn.")
    else:
        with st.spinner("Đang tư vấn... 🤖"):
            result = rag_chain(user_pref)

        # Hiển thị kết quả
        st.subheader("👉 Ngành học gợi ý:")
        st.markdown(result['result'])