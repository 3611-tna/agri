import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
from io import BytesIO

# ==============================
# ⚙️ Cấu hình giao diện
# ==============================
st.set_page_config(
    page_title="💡 AgriAnalyze Khách hàng Agribank",
    layout="wide"
)

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==============================
# 🎨 Tiêu đề ứng dụng
# ==============================
st.markdown("""
<h1 style='text-align:center; color:#AE1C3F;'>💡 AGRIANALYZE – TƯ VẤN KHÁCH HÀNG AGRIBANK</h1>
<p style='text-align:center; color:gray;'>Phân tích hành vi, tính cách và tư vấn sản phẩm phù hợp cho khách hàng Agribank</p>
<hr style='border:1px solid #AE1C3F'>
""", unsafe_allow_html=True)

# ==============================
# 🔑 Nhập API Key Gemini
# ==============================
with st.expander("🔑 Cấu hình API Key Gemini"):
    api_key = st.text_input("Nhập API Key của bạn:", type="password", placeholder="AIzaSy...")
    st.caption("API Key chỉ được dùng cục bộ, không lưu lại trên máy chủ.")

# ==============================
# 📤 Upload file Excel khách hàng
# ==============================
uploaded_file = st.file_uploader(
    "📂 Tải file Excel dữ liệu khách hàng Agribank (sheet: KhachHang)",
    type=["xlsx", "xls"]
)

# ==============================
# 🤖 Hàm gọi Gemini AI
# ==============================
def get_ai_analysis(customer_info, api_key):
    try:
        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash"

        prompt = f"""
        Bạn là một chuyên gia tư vấn khách hàng của Agribank.
        Hãy phân tích và tư vấn chi tiết cho khách hàng dựa trên thông tin sau:

        {customer_info}

        Yêu cầu phản hồi gồm:
        1. Đánh giá tổng quan về khách hàng (hành vi, tính cách, sở thích).
        2. Phân tích theo độ tuổi, nghề nghiệp, thu nhập, tôn giáo, chính trị, cung hoàng đạo (nếu có).
        3. Dự đoán hành vi tài chính và nhu cầu sản phẩm ngân hàng.
        4. Gợi ý sản phẩm/dịch vụ phù hợp của Agribank (tiền gửi, tín dụng, bảo hiểm, số hoá...).
        5. Đưa ra khuyến nghị tương tác / chăm sóc phù hợp.

        Viết ngắn gọn, súc tích, chuyên nghiệp, tối đa 5 đoạn.
        """

        response = client.models.generate_content(model=model, contents=prompt)
        return response.text
    except APIError as e:
        return f"⚠️ Lỗi gọi Gemini API: {e}"
    except Exception as e:
        return f"⚠️ Lỗi khác: {e}"

# ==============================
# 📊 Xử lý và hiển thị dữ liệu
# ==============================
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="KhachHang")

        st.success(f"✅ Đã tải {len(df)} khách hàng từ file Excel.")

        # Hiển thị danh sách khách hàng
        customer_list = df["Họ tên"].tolist()
        selected_customers = st.multiselect(
            "👥 Chọn khách hàng cần phân tích:",
            options=customer_list,
            default=customer_list[:1] if len(customer_list) > 0 else []
        )

        if len(selected_customers) > 0:
            selected_df = df[df["Họ tên"].isin(selected_customers)]

            st.dataframe(selected_df, use_container_width=True)

            if st.button("🚀 Phân tích & Tư vấn bằng Gemini AI"):
                if not api_key:
                    st.error("❌ Vui lòng nhập API Key Gemini trước.")
                else:
                    results = []
                    for _, row in selected_df.iterrows():
                        customer_info = row.to_dict()
                        analysis = get_ai_analysis(customer_info, api_key)
                        results.append({
                            "Họ tên": row["Họ tên"],
                            "Phân tích & tư vấn": analysis
                        })
                    results_df = pd.DataFrame(results)

                    st.subheader("🧠 Kết quả phân tích & tư vấn")
                    for i, row in results_df.iterrows():
                        st.markdown(f"### 👤 {row['Họ tên']}")
                        st.info(row["Phân tích & tư vấn"])

                    # Xuất Excel
                    output = BytesIO()
                    results_df.to_excel(output, index=False, engine="openpyxl")
                    st.download_button(
                        label="⬇️ Tải kết quả tư vấn (Excel)",
                        data=output.getvalue(),
                        file_name="tu_van_khach_hang.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    except Exception as e:
        st.error(f"⚠️ Lỗi khi đọc file Excel: {e}")
else:
    st.info("⬆️ Vui lòng tải lên file khách hàng để bắt đầu phân tích.")

# ==============================
# 🔚 Footer
# ==============================
st.markdown("""
<footer style='text-align:center; margin-top:40px; padding:10px; background:#AE1C3F; color:white; border-radius:12px;'>
© 2025 Agribank Training & Development — Ứng dụng <b>AgriAnalyze Khách hàng</b><br>
Phát triển bởi Bộ phận CNTT Agribank
</footer>
""", unsafe_allow_html=True)
