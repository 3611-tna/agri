import streamlit as st
import pandas as pd
import concurrent.futures
from io import BytesIO
from google import genai

# --- OPENAI Import ---
from openai import OpenAI
import openai

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="🤖 AgriAI CRM Pro", layout="wide", page_icon="🤖")

# ================== CUSTOM STYLE =================
with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🤖 AGRIAI CRM PRO</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Phân tích & tư vấn khách hàng Agribank bằng AI (OpenAI | Gemini | Hybrid)</p><hr>", unsafe_allow_html=True)

# ================== CONFIG PANEL =================
with st.expander("⚙️ Cấu hình AI"):
    c1, c2 = st.columns(2)
    with c1:
        openai_key = st.text_input("🔹 OpenAI API Key", type="password")
        openai_model = st.selectbox("🔹 Model OpenAI", ["gpt-4o-mini", "gpt-4-turbo", "gpt-5"])
    with c2:
        gemini_key = st.text_input("🔸 Gemini API Key", type="password")
        gemini_model = st.selectbox("🔸 Model Gemini", ["gemini-2.0-flash", "gemini-1.5-flash"])
    creativity = st.slider("🎨 Mức sáng tạo (0–2)", 0.0, 2.0, 1.0, 0.1)
    ai_mode = st.radio("🤝 Chế độ AI", ["Gemini", "OpenAI", "Hybrid"], horizontal=True)

# ================== FILE UPLOAD ==================
uploaded = st.file_uploader("📥 Tải file Excel khách hàng (sheet: KhachHang)", type=["xlsx", "xls"])

# ================== AI FUNCTIONS ==================
def call_gemini(prompt, key, model_name):
    """Gọi Gemini API"""
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=model_name, contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini lỗi: {e}"

def call_openai(prompt, key, model_name, creativity):
    """Gọi OpenAI API (đã fix proxies hoàn toàn)"""
    try:
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia phân tích khách hàng Agribank có 15 năm kinh nghiệm."},
                {"role": "user", "content": prompt},
            ],
            temperature=creativity,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # fallback nếu version openai cũ
        try:
            openai.api_key = key
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=creativity
            )
            return resp.choices[0].message["content"].strip()
        except Exception as inner_e:
            return f"⚠️ OpenAI lỗi: {inner_e}"

def export_excel(customer_name, analysis_text):
    df = pd.DataFrame({"Khách hàng": [customer_name], "Phân tích & Tư vấn": [analysis_text]})
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="PhanTich", index=False)
    return buf.getvalue()

# ================== MEMORY INIT ==================
if "memory" not in st.session_state:
    st.session_state["memory"] = {}

# ================== MAIN PROCESS ==================
if uploaded:
    df = pd.read_excel(uploaded, sheet_name="KhachHang")
    st.success(f"✅ Đã tải {len(df)} khách hàng trong danh sách.")

    search_term = st.text_input("🔍 Nhập tên khách hàng để tìm kiếm...")
    filtered_df = df[df["Họ tên"].str.contains(search_term, case=False, na=False)] if search_term else df

    st.markdown("<h4>📋 Danh sách khách hàng</h4>", unsafe_allow_html=True)
    st.dataframe(
        filtered_df.style.highlight_max(axis=0, color="#fde8eb"),
        use_container_width=True
    )

    selected = st.selectbox("👥 Chọn khách hàng cần phân tích", filtered_df["Họ tên"].tolist())

    if selected not in st.session_state["memory"]:
        st.session_state["memory"][selected] = {"chat": [], "missing": []}

    cust_data = df[df["Họ tên"] == selected].to_dict(orient="records")[0]

    if st.button("🚀 Phân tích khách hàng"):
        missing = [col for col, val in cust_data.items() if pd.isna(val) or val == ""]
        st.session_state["memory"][selected]["missing"] = missing
        st.info(f"🧩 Dữ liệu thiếu: {', '.join(missing) if missing else 'Không có'}")

        prompt = f"""
        Bạn là chuyên gia Agribank có 15 năm kinh nghiệm.
        Hãy phân tích khách hàng này dưới 4 góc độ:
        1️⃣ Năng lực tài chính & hành vi giao dịch.
        2️⃣ Tâm lý tiêu dùng, sở thích, độ tuổi, tôn giáo, khu vực.
        3️⃣ Sản phẩm/dịch vụ nên tiếp cận & chiến lược chăm sóc.
        4️⃣ Định hướng duy trì và gia tăng quan hệ lâu dài.
        Dữ liệu khách hàng: {cust_data}
        """

        if ai_mode == "Hybrid":
            with concurrent.futures.ThreadPoolExecutor() as pool:
                futures = []
                if gemini_key:
                    futures.append(pool.submit(call_gemini, prompt, gemini_key, gemini_model))
                if openai_key:
                    futures.append(pool.submit(call_openai, prompt, openai_key, openai_model, creativity))
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
                summary = "\n\n---\n\n".join(results)
        elif ai_mode == "Gemini":
            summary = call_gemini(prompt, gemini_key, gemini_model)
        else:
            summary = call_openai(prompt, openai_key, openai_model, creativity)

        st.session_state["memory"][selected]["analysis"] = summary
        st.markdown(f"<div class='analysis-card'>{summary}</div>", unsafe_allow_html=True)

        excel_data = export_excel(selected, summary)
        st.download_button("📊 Tải kết quả (Excel)", excel_data, file_name=f"{selected}_phan_tich.xlsx")

    # ====== CHAT BỔ SUNG ======
    st.markdown("<h3>💬 Trợ lý AI - Bổ sung dữ liệu hoặc hỏi thêm</h3>", unsafe_allow_html=True)
    for msg in st.session_state["memory"][selected]["chat"]:
        role = msg["role"]
        icon = "🧑‍💼" if role == "user" else "🤖"
        color = "#AE1C3F" if role == "user" else "#333"
        st.markdown(f"<div class='chat-msg' style='border-left:3px solid {color};'><b>{icon} {role.title()}:</b> {msg['content']}</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Nhập câu hỏi hoặc thông tin bổ sung...")
    if user_input:
        st.session_state["memory"][selected]["chat"].append({"role": "user", "content": user_input})
        missing = st.session_state["memory"][selected]["missing"]

        prompt = f"""
        Dữ liệu khách hàng: {cust_data}
        Nhân viên Agribank vừa nói: "{user_input}".
        Các thông tin còn thiếu: {missing}.
        Hãy phản hồi phù hợp, chỉ hỏi thêm khi thật sự cần thiết.
        """
        response = call_openai(prompt, openai_key, openai_model, creativity)
        st.session_state["memory"][selected]["chat"].append({"role": "assistant", "content": response})

        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
else:
    st.info("⬆️ Vui lòng tải file Excel khách hàng trước khi bắt đầu.")

# ========== FOOTER ==========
st.markdown("""
<div class='footer'>
    © 2025 <span>Agribank AI Division</span> | <b>AgriAI CRM Pro v2.7.4</b><br>
    <i>Ứng dụng hỗ trợ cán bộ Agribank trong công tác phân tích & tư vấn khách hàng.</i>
</div>
""", unsafe_allow_html=True)
