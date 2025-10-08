# ======================================================
# 🌾 AgriAI CRM PRO 4.3.0 – Chat AI Edition
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from google import genai
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import base64

st.set_page_config(page_title="AgriAI CRM Pro 4.3.0", layout="wide", page_icon="🌾")

# Khởi tạo session state cho chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_customer_context" not in st.session_state:
    st.session_state.current_customer_context = None
if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = ""

# ---------- GIAO DIỆN ----------
st.markdown("""
<style>
h1,h2,h3,h4 {color:#AE1C3F;}
.stButton>button {
    background:linear-gradient(90deg,#AE1C3F,#D72638);
    color:white;font-weight:bold;border:none;
    border-radius:8px;padding:0.5em 1.2em;
}
.stButton>button:hover {opacity:0.9;transform:scale(1.03);}
.analysis-box {
    background:white;padding:18px;border-radius:10px;
    border-left:6px solid #AE1C3F;
    box-shadow:0 2px 10px rgba(0,0,0,0.1);
    margin-bottom:25px;
}
.chat-message {
    padding:12px;border-radius:8px;margin-bottom:10px;
}
.user-message {
    background:#f0f0f0;border-left:4px solid #AE1C3F;
}
.ai-message {
    background:#fff5f7;border-left:4px solid #D72638;
}
.footer {text-align:center;color:#777;margin-top:40px;}
</style>
""", unsafe_allow_html=True)

# Hiển thị logo
try:
    with open("logo.png", "rb") as f:
        logo_data = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<img src="data:image/png;base64,{logo_data}" style="width:180px;margin-bottom:20px;">',
            unsafe_allow_html=True
        )
except:
    pass

st.title("🌾 AgriAI CRM PRO 4.3.0 – AI Chat & Phân tích thông minh")
st.caption("Tự động phân tích khách hàng, dự báo xu hướng & chat với AI để tư vấn chuyên sâu")

# ---------- CẤU HÌNH ----------
with st.expander("⚙️ Cấu hình Gemini"):
    gemini_key = st.text_input("🔸 Gemini API Key", type="password")
    gemini_model = st.selectbox("🔹 Model Gemini", ["gemini-2.0-flash", "gemini-1.5-flash"])
    creativity = st.slider("🎨 Mức sáng tạo (0 - 2)", 0.0, 2.0, 0.8, 0.1)

uploaded = st.file_uploader("📥 Tải file Excel (sheet KhachHang)", type=["xlsx"])

# ---------- HÀM CỐT LÕI ----------
def call_gemini(prompt, key, model, creativity):
    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=model, contents=prompt, config={"temperature":creativity})
        return resp.text.strip()
    except Exception as e:
        return f"⚠️ Lỗi Gemini: {e}"

def call_gemini_with_history(user_msg, key, model, creativity, context="", history=[]):
    """Gọi Gemini với lịch sử hội thoại và ngữ cảnh"""
    try:
        client = genai.Client(api_key=key)
        
        # Xây dựng prompt với ngữ cảnh và lịch sử
        system_prompt = f"""Bạn là chuyên viên tư vấn khách hàng Agribank có 15 năm kinh nghiệm, 
đang hỗ trợ cán bộ tín dụng phân tích và tư vấn khách hàng.

{context}

Hãy trả lời câu hỏi dựa trên thông tin đã có và lịch sử hội thoại. 
Nếu cần thêm thông tin, hãy hỏi cụ thể. Trả lời ngắn gọn, chuyên nghiệp và thực tiễn."""

        # Kết hợp lịch sử
        full_prompt = system_prompt + "\n\n"
        for msg in history[-6:]:  # Lấy 6 tin nhắn gần nhất
            full_prompt += f"{msg['role']}: {msg['content']}\n"
        full_prompt += f"User: {user_msg}\nAssistant:"
        
        resp = client.models.generate_content(
            model=model, 
            contents=full_prompt, 
            config={"temperature": creativity}
        )
        return resp.text.strip()
    except Exception as e:
        return f"⚠️ Lỗi Gemini: {e}"

def calc_scores(row):
    try:
        thu_nhap = float(str(row.get("Thu nhập", 0)).replace(",", "").replace(".", ""))
        gui = float(str(row.get("Số dư tiền gửi", 0)).replace(",", "").replace(".", ""))
        vay = float(str(row.get("Số dư tiền vay", 0)).replace(",", "").replace(".", ""))

        rui_ro = min(100, max(0, (vay / (thu_nhap + 1)) * 60))
        tiem_nang = min(100, (thu_nhap + gui) / 1_000_000 * 10)
        gan_bo = 40 + len(str(row.get("Dịch vụ đang dùng", "")).split(",")) * 10
        if "Thành phố" in str(row.get("Khu vực", "")): gan_bo += 10

        return round(rui_ro,1), round(tiem_nang,1), round(gan_bo,1)
    except:
        return 0,0,0

def suggest_action(r, t, g, churn_prob, loan_prob):
    acts = []
    if churn_prob > 0.7:
        acts.append("⚠️ Nguy cơ rời bỏ cao → cần CSKH hoặc ưu đãi duy trì.")
    if loan_prob > 0.7:
        acts.append("💰 Tiềm năng vay thêm → gợi ý gói vay tiêu dùng / sản xuất.")
    if t > 80:
        acts.append("📈 Đề xuất gói tiết kiệm linh hoạt hoặc đầu tư dài hạn.")
    if g < 50:
        acts.append("💬 Tăng tương tác qua chương trình tri ân / CSKH định kỳ.")
    if not acts:
        acts.append("✅ Khách hàng ổn định, duy trì chăm sóc định kỳ.")
    return acts

def export_excel(name, summary, df_scores):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame({"Khách hàng":[name], "Phân tích":[summary]}).to_excel(writer, sheet_name="PhanTich", index=False)
        df_scores.to_excel(writer, sheet_name="Scores", index=False)
    return out.getvalue()

def train_predictive_model(df):
    scaler = MinMaxScaler()
    X = df[["Điểm Rủi ro","Điểm Tiềm năng","Điểm Gắn bó"]]
    X_scaled = scaler.fit_transform(X)

    churn_labels = np.where(df["Điểm Gắn bó"] < 50, 1, 0)
    loan_labels = np.where(df["Điểm Tiềm năng"] > 70, 1, 0)

    # Nếu dữ liệu có <2 lớp, tạo nhãn giả để tránh lỗi
    if len(np.unique(churn_labels)) < 2:
        churn_labels[0] = 1
    if len(np.unique(loan_labels)) < 2:
        loan_labels[-1] = 1

    churn_model = LogisticRegression().fit(X_scaled, churn_labels)
    loan_model = LogisticRegression().fit(X_scaled, loan_labels)

    return churn_model, loan_model, scaler

# ---------- CHƯƠNG TRÌNH CHÍNH ----------
if uploaded:
    df = pd.read_excel(uploaded, sheet_name="KhachHang")
    st.success(f"✅ Đã tải {len(df)} khách hàng.")
    scores = [calc_scores(row) for _, row in df.iterrows()]
    df["Điểm Rủi ro"], df["Điểm Tiềm năng"], df["Điểm Gắn bó"] = zip(*scores)
    st.dataframe(df, use_container_width=True)

    # Huấn luyện mô hình (ổn định)
    churn_model, loan_model, scaler = train_predictive_model(df)

    selected = st.selectbox("👤 Chọn khách hàng để phân tích", df["Họ tên"].tolist())
    cust = df[df["Họ tên"] == selected].iloc[0].to_dict()

    if st.button("🚀 Phân tích chuyên sâu & dự báo hành động"):
        r,t,g = calc_scores(cust)
        features = np.array([[r,t,g]])
        X_scaled = scaler.transform(features)
        churn_prob = churn_model.predict_proba(X_scaled)[0][1]
        loan_prob = loan_model.predict_proba(X_scaled)[0][1]

        actions = suggest_action(r,t,g,churn_prob,loan_prob)

        prompt = f"""
Bạn là chuyên viên tư vấn khách hàng Agribank có 15 năm kinh nghiệm.
Dưới đây là hồ sơ khách hàng:
{cust}

Điểm hệ thống:
- Rủi ro: {r}
- Tiềm năng: {t}
- Gắn bó: {g}
- Xác suất rời bỏ: {round(churn_prob*100,2)}%
- Xác suất vay thêm: {round(loan_prob*100,2)}%

Hãy viết bản PHÂN TÍCH NGHIỆP VỤ gồm:
1️⃣ Tổng quan tài chính & hành vi khách hàng.
2️⃣ Rủi ro & xu hướng hành động.
3️⃣ Gợi ý sản phẩm Agribank phù hợp.
4️⃣ Kế hoạch hành động cụ thể (trong 1–3 tháng).

Ngắn gọn, rõ ràng, chuyên nghiệp và thực tiễn.
"""
        ai_text = call_gemini(prompt, gemini_key, gemini_model, creativity)

        # Lưu ngữ cảnh khách hàng cho chat
        st.session_state.current_customer_context = cust
        st.session_state.analysis_context = f"""
THÔNG TIN KHÁCH HÀNG ĐANG PHÂN TÍCH:
{cust}

ĐIỂM HỆ THỐNG:
- Rủi ro: {r}
- Tiềm năng: {t}
- Gắn bó: {g}
- Xác suất rời bỏ: {round(churn_prob*100,2)}%
- Xác suất vay thêm: {round(loan_prob*100,2)}%

PHÂN TÍCH AI:
{ai_text}

HÀNH ĐỘNG ĐỀ XUẤT:
{chr(10).join(actions)}
"""

        summary = f"""
### 📌 Phân tích khách hàng **{selected}**
#### 🔢 Điểm hệ thống
- Rủi ro: {r} | Tiềm năng: {t} | Gắn bó: {g}
- Xác suất rời bỏ: {round(churn_prob*100,2)}% | Xác suất vay thêm: {round(loan_prob*100,2)}%

#### 💡 Nhận định chuyên gia (AI)
{ai_text}

#### 🎯 Hành động đề xuất
{chr(10).join(actions)}
"""
        st.markdown(f"<div class='analysis-box'>{summary}</div>", unsafe_allow_html=True)

        df_scores = pd.DataFrame({
            "Chỉ tiêu":["Rủi ro","Tiềm năng","Gắn bó","Xác suất rời bỏ","Xác suất vay thêm"],
            "Giá trị":[r,t,g,round(churn_prob*100,2),round(loan_prob*100,2)]
        })
        excel = export_excel(selected, summary, df_scores)
        st.download_button("📊 Tải báo cáo chi tiết (Excel)", excel, file_name=f"{selected}_AgriAI_Insight.xlsx")

    # ========== PHẦN CHAT AI ==========
    st.markdown("---")
    st.subheader("💬 Chat với AI Tư vấn")
    
    if st.session_state.analysis_context:
        st.info("✅ AI đã nắm thông tin khách hàng. Bạn có thể hỏi thêm về phân tích, đề xuất hành động hoặc cung cấp thông tin bổ sung.")
    else:
        st.warning("⚠️ Hãy phân tích một khách hàng trước để AI có đủ ngữ cảnh tư vấn.")
    
    # Hiển thị lịch sử chat
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "User":
                st.markdown(f"<div class='chat-message user-message'><b>🧑 Bạn:</b> {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message ai-message'><b>🤖 AI:</b> {msg['content']}</div>", unsafe_allow_html=True)
    
    # Input chat
    col1, col2 = st.columns([5,1])
    with col1:
        user_input = st.text_input("💭 Nhập câu hỏi hoặc thông tin bổ sung:", key="chat_input", placeholder="VD: Khách hàng này có quan hệ tốt với chi nhánh không?")
    with col2:
        send_btn = st.button("Gửi", use_container_width=True)
        clear_btn = st.button("Xóa lịch sử", use_container_width=True)
    
    if clear_btn:
        st.session_state.chat_history = []
        st.rerun()
    
    if send_btn and user_input.strip():
        # Thêm tin nhắn của user
        st.session_state.chat_history.append({"role": "User", "content": user_input})
        
        # Gọi AI với context và history
        ai_response = call_gemini_with_history(
            user_input, 
            gemini_key, 
            gemini_model, 
            creativity,
            context=st.session_state.analysis_context,
            history=st.session_state.chat_history
        )
        
        # Thêm phản hồi AI
        st.session_state.chat_history.append({"role": "Assistant", "content": ai_response})
        st.rerun()

else:
    st.info("⬆️ Hãy tải file Excel khách hàng (sheet KhachHang) để bắt đầu phân tích.")

st.markdown("<div class='footer'>© 2025 Agribank AI | AgriAI CRM Pro 4.3.0 - Chat AI Edition</div>", unsafe_allow_html=True)
