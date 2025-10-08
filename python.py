import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini (Được điều chỉnh để dùng cho Chat) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm Xử lý Chat tương tác ---
def chat_with_gemini(prompt, api_key, context_data=None):
    """Xử lý logic chat, bao gồm lịch sử hội thoại và ngữ cảnh dữ liệu."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 
        
        # 1. Khởi tạo Chat nếu chưa có
        if "chat_session" not in st.session_state:
            # Tạo prompt hệ thống ban đầu với ngữ cảnh phân tích
            system_instruction = "Bạn là một trợ lý AI thông minh chuyên về phân tích tài chính. Hãy trả lời các câu hỏi của người dùng một cách chính xác, ngắn gọn và dựa trên ngữ cảnh dữ liệu tài chính mà họ đã cung cấp. Nếu không có dữ liệu, hãy hỏi lại."
            if context_data:
                 system_instruction += f"\n\nDữ liệu Phân tích Hiện tại:\n{context_data}\n\n"
            
            # Sử dụng Chat API để duy trì lịch sử
            st.session_state.chat_session = client.chats.create(
                model=model_name,
                system_instruction=system_instruction
            )

        # 2. Gửi tin nhắn đến Chat Session
        response = st.session_state.chat_session.send_message(prompt)
        return response.text

    except APIError as e:
        return f"Lỗi API: Vui lòng kiểm tra Khóa API. Chi tiết: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định trong phiên chat: {e}"

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Khởi tạo state cho Lịch sử Chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if uploaded_file is not None:
    # ... (giữ nguyên logic xử lý file) ...
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())
        
        # --- Chức năng 4 & 5: Tính Chỉ số & Chuẩn bị Nhận xét AI ---
        thanh_toan_hien_hanh_N = "N/A"
        thanh_toan_hien_hanh_N_1 = "N/A"
        data_for_ai = ""

        if df_processed is not None:
            # ********** (Logic Hiển thị Bảng Dữ liệu và Metric được giữ nguyên) **********
            
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành (Ví dụ)
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                
                # ****** ĐÃ SỬA LỖI TYPO TẠI ĐÂY ******
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                # *************************************
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"
                
            # Chuẩn bị dữ liệu để gửi cho AI (Dùng cho cả Chức năng 5 và Chat)
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if 'TÀI SẢN NGẮN HẠN' in df_processed['Chỉ tiêu'].str.upper().values else "N/A",
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            # --- Chức năng 5: Nhận xét AI Tự động ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI Tự động)")
            
            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        # Giữ nguyên khối này, nhưng lỗi typo đã được khắc phục ở trên
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
    data_for_ai = "" # Đảm bảo biến này tồn tại cho phần chat

# ==============================================================================
#                 PHẦN KHUNG CHAT (Đặt trong Sidebar)
# ==============================================================================
st.sidebar.title("💬 Trợ lý Chat Tài chính (Gemini)")
st.sidebar.markdown("Sử dụng khung chat này để đặt câu hỏi chi tiết về dữ liệu tài chính bạn vừa tải lên.")

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.sidebar.chat_message(message["role"]):
        st.markdown(message["content"])

# Khung nhập liệu cho Chat
if prompt := st.sidebar.chat_input("Bạn muốn hỏi gì về báo cáo này?"):
    
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.sidebar.error("Lỗi: Vui lòng cấu hình Khóa API 'GEMINI_API_KEY' để sử dụng chức năng chat.")
    else:
        # Thêm tin nhắn người dùng vào lịch sử
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Hiển thị tin nhắn người dùng
        with st.sidebar.chat_message("user"):
            st.markdown(prompt)

        # Xử lý và hiển thị phản hồi của AI
        with st.sidebar.chat_message("assistant"):
            with st.spinner("Đang hỏi Gemini..."):
                # Gọi hàm chat, truyền dữ liệu phân tích làm ngữ cảnh
                full_context = f"Dữ liệu Phân tích Tổng hợp: {data_for_ai}" if uploaded_file is not None else "Không có dữ liệu tài chính được tải lên."
                
                # Gọi hàm chat_with_gemini
                response = chat_with_gemini(prompt, api_key, full_context)
                
                # Hiển thị phản hồi
                st.markdown(response)

            # Thêm phản hồi AI vào lịch sử
            st.session_state.messages.append({"role": "assistant", "content": response})

# Nút reset chat session (Tùy chọn)
if st.sidebar.button("Bắt đầu Phiên Chat Mới"):
    st.session_state["messages"] = []
    if "chat_session" in st.session_state:
        del st.session_state["chat_session"]
    st.rerun()
