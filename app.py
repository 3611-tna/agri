import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# ==============================
# ⚙️ Cấu hình giao diện chính
# ==============================
st.set_page_config(
    page_title="📊 Phân Tích Báo Cáo Tài Chính - AgriAnalyze Pro",
    layout="wide"
)

# Load CSS & Theme
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==============================
# 🎨 Tiêu đề chính
# ==============================
st.markdown("""
<h1 style='text-align:center; color:#AE1C3F;'>📊 AGRIANALYZE PRO</h1>
<p style='text-align:center; color:gray;'>Ứng dụng phân tích báo cáo tài chính Agribank bằng AI</p>
<hr style='border:1px solid #AE1C3F'>
""", unsafe_allow_html=True)

# ==============================
# 🔑 Nhập API Key Gemini
# ==============================
with st.expander("🔑 Cấu hình API Key Gemini"):
    api_key = st.text_input("Nhập API Key của bạn:", type="password", placeholder="Ví dụ: AIzaSy...")
    st.caption("👉 API Key chỉ được dùng cục bộ, không lưu trên server.")

# ==============================
# 📤 Upload file Excel
# ==============================
uploaded_file = st.file_uploader(
    "📂 Tải file Excel Báo cáo Tài chính (cột: Chỉ tiêu | Năm trước | Năm sau)",
    type=["xlsx", "xls"]
)

# ==============================
# 🧮 Hàm xử lý dữ liệu
# ==============================
@st.cache_data
def process_financial_data(df):
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    tong_tai_san = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    if tong_tai_san.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    divisor_N_1 = tong_tai_san['Năm trước'].iloc[0] if tong_tai_san['Năm trước'].iloc[0] != 0 else 1e-9
    divisor_N = tong_tai_san['Năm sau'].iloc[0] if tong_tai_san['Năm sau'].iloc[0] != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    return df

# ==============================
# 🤖 Hàm gọi Gemini AI
# ==============================
def get_ai_analysis(data_for_ai, api_key):
    try:
        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash"
        prompt = f"""
        Bạn là chuyên gia tài chính. Dựa vào dữ liệu sau, hãy nhận xét ngắn gọn (3–4 đoạn)
        về tình hình tài chính: tốc độ tăng trưởng, cơ cấu tài sản, thanh toán hiện hành.

        Dữ liệu:
        {data_for_ai}
        """
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text
    except APIError as e:
        return f"⚠️ Lỗi khi gọi Gemini API: {e}"
    except Exception as e:
        return f"⚠️ Lỗi khác: {e}"

# ==============================
# 📊 Phân tích dữ liệu
# ==============================
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        df_result = process_financial_data(df.copy())

        st.subheader("📈 Tốc độ Tăng trưởng & Tỷ trọng Tài sản")
        st.dataframe(
            df_result.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }),
            use_container_width=True
        )

        # Tính toán chỉ số thanh toán
        try:
            tsnh_n = df_result[df_result['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False)]['Năm sau'].iloc[0]
            tsnh_n_1 = df_result[df_result['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False)]['Năm trước'].iloc[0]
            no_ngan_han_N = df_result[df_result['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False)]['Năm sau'].iloc[0]
            no_ngan_han_N_1 = df_result[df_result['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False)]['Năm trước'].iloc[0]

            ratio_n = tsnh_n / no_ngan_han_N
            ratio_n_1 = tsnh_n_1 / no_ngan_han_N_1

            col1, col2 = st.columns(2)
            col1.metric("Thanh toán Hiện hành (Năm trước)", f"{ratio_n_1:.2f} lần")
            col2.metric("Thanh toán Hiện hành (Năm sau)", f"{ratio_n:.2f} lần", delta=f"{ratio_n - ratio_n_1:.2f}")
        except:
            st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN'.")

        # Phân tích AI
        st.subheader("🧠 Phân tích AI – Nhận xét Tình hình Tài chính")
        data_for_ai = df_result.to_markdown(index=False)

        if st.button("🚀 Gửi dữ liệu đến Gemini AI"):
            if not api_key:
                st.error("❌ Vui lòng nhập API Key Gemini.")
            else:
                with st.spinner("🔍 Đang phân tích..."):
                    result = get_ai_analysis(data_for_ai, api_key)
                    st.info(result)

        # Xuất kết quả ra Excel
        st.download_button(
            "⬇️ Tải kết quả Excel",
            data=df_result.to_excel(index=False, engine='openpyxl'),
            file_name="bao_cao_phan_tich.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"⚠️ Lỗi khi xử lý file: {e}")
else:
    st.info("⬆️ Vui lòng tải lên file Excel để bắt đầu phân tích.")

# ==============================
# 🔚 Footer
# ==============================
st.markdown("""
<footer style='text-align:center; margin-top:40px; padding:10px; background:#AE1C3F; color:white; border-radius:12px;'>
© 2025 Agribank Training & Development — Ứng dụng <b>AgriAnalyze Pro</b><br>
Phát triển bởi Bộ phận CNTT Agribank
</footer>
""", unsafe_allow_html=True)
