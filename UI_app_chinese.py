import asyncio  # 必須在最前面
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import base64
from datetime import datetime

import signal
import threading
import time
import sys

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]


# 紀錄當前 Python 程式的 Process ID
pid = os.getpid()
CHECK_INTERVAL = 5  # 每 5 秒檢查一次 (避免太頻繁影響效能)

# 檢查 WebSocket 是否仍然活著的函式
def check_if_closed():
    # 等待 Streamlit 完整啟動後再開始檢查
    time.sleep(10)  # 等待 10 秒後啟動檢查機制
    while True:
        try:
            # 使用 asyncio 確認 Streamlit 是否仍然運行
            asyncio.run(asyncio.sleep(0))  # 如果這行能執行，表示 Streamlit 還在運行中
        except:
            # 如果檢測到程序無法正常運行，強制結束
            os.kill(pid, signal.SIGTERM)
            sys.exit()  
        time.sleep(CHECK_INTERVAL)  # 每 5 秒檢查一次

# 啟動背景檢查執行緒
threading.Thread(target=check_if_closed, daemon=True).start()



# 定義轉換圖片為 base64 編碼字串的函數

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# 取得圖片的 base64 字串
lab_logo = get_image_base64('./logo_lab.png')
hos_logo = get_image_base64('./logo_hos.png')

# 設定頁面標題
st.set_page_config(page_title='皮膚鏡影像分類系統', layout='wide')

# 樣式設定
st.markdown(
    '''
    <style>
    .header-bar {
        background-color: #E0F0FF;
        padding: 0;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        left: 0;
        right: 0;
        height: 130px;
        z-index: 9999;
        border-bottom: 1px solid #ddd;
        width: 100%;
    }
    .header-content {
        display: flex;
        align-items: center;
    }
    .header-bar h1 {
        font-size: 32px;
        margin: 0;
        font-style: italic;
        text-align: center;
        flex-grow: 1;
        font-family: 'Arial', sans-serif;
        color: #0047AB;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 14px;
        color: #888;
    }
    .prediction-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 7px;
        text-align: center;
        font-size: 24px;
        background-color: #F8F8F8;
        transition: all 0.3s;
    }
    .highlight-box {
        border-radius: 10px;
        padding: 10px;
        margin: 7px;
        text-align: center;    
        background-color: #C1FFC1;
        border: 2px solid #4CAF50;
        font-size: 28px;
        transform: scale(1.03);
        font-weight: bold;
        transition: all 0.3s;
    }
    .grid-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
    }
    /* 小螢幕適應 */
    @media (max-width: 1200px) {
        .grid-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 800px) {
        .grid-container {
            grid-template-columns: repeat(1, 1fr);
        }
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# 網頁頂部標題欄
st.markdown(
    f'''
    <div class="header-bar">
        <div class="header-content">
            <img src="data:image/png;base64,{lab_logo}" alt="Lab Logo" style="height: 125px; margin-left: 20px;">
        </div>
        <h1>Dermoscopy Diagnosis Tool</h1>
        <div class="header-content">
            <img src="data:image/png;base64,{hos_logo}" alt="Hospital Logo" style="height: 125px; margin-right: 20px;">
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)

# 增加空白區域讓標題不被蓋住
st.markdown('<div style="margin-top: 110px;"></div>', unsafe_allow_html=True)

# 設定資料夾路徑
UPLOAD_DIR = '../Diagnostic Folder/'

if os.path.exists(UPLOAD_DIR) == False:
    os.mkdir(UPLOAD_DIR)
    
# 處理檔案上傳
uploaded_files = st.sidebar.file_uploader('批量上傳影像', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 0:
        # 建立新資料夾，以年月日時為名稱
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_folder = os.path.join(UPLOAD_DIR, timestamp)
        os.makedirs(new_folder, exist_ok=True)

        # 將每個檔案儲存到該資料夾
        for uploaded_file in uploaded_files:
            image_path = os.path.join(new_folder, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.sidebar.success(f'檔案已上傳至資料夾: {timestamp}')



MODEL_PATH = "./best.pt"  # 修改成你的模型路徑

# 載入模型
@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

# 預測函數
def predict(image_path):
    results = model.predict(image_path, verbose=False)
    predict_proba = results[0].probs.data.cpu().numpy()
    prediction_dict = {
        'SCC': predict_proba[0],
        'BCC': predict_proba[1],
        'BKL': predict_proba[2],
        'DF': predict_proba[3],
        'MEL': predict_proba[4],
        'NV': predict_proba[5],
        'VASC': predict_proba[6],
    }
    return prediction_dict

disease_display_names = {
    'SCC': '鱗狀細胞癌(惡性)',
    'BCC': '基底細胞癌(惡性)',
    'BKL': '良性角化病變(良性)',
    'DF': '皮膚纖維瘤(良性)',
    'MEL': '黑色素瘤(惡性)',
    'NV': '痣(良性)',
    'VASC': '血管病變(良性)'
}


# 設定類別名稱
classes = ['SCC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

# 左側邊欄 - 資料夾選擇
st.sidebar.title('影像檢視與選擇')

# 取得資料夾清單
folders = [f for f in os.listdir(UPLOAD_DIR) if os.path.isdir(os.path.join(UPLOAD_DIR, f))]
selected_folder = st.sidebar.selectbox('選擇資料夾', folders)

# 根據選擇的資料夾取得影像檔案
image_files = []
if selected_folder:
    folder_path = os.path.join(UPLOAD_DIR, selected_folder)
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 影像檔案選擇
selected_image = st.sidebar.selectbox('選擇影像', image_files)

# 讀取選擇的影像
if selected_image:
    image_path = os.path.join(folder_path, selected_image)
    image = Image.open(image_path)

# 主區域 - 影像顯示與按鈕
col1, col2 = st.columns([1.7, 1.3])

with col1:
    if selected_image:
        st.image(image, use_container_width=True)
        
        st.markdown(
            '''
            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #6c757d; margin-top: 10px;">
                選擇的影像
            </div>
            ''', 
            unsafe_allow_html=True
        )


        st.markdown(
            """
            <style>
            div.stButton > button {
                font-size: 24px; /* 字體大小 */
                padding: 10px 20px; /* 上下與左右的內邊距 */
                border-radius: 8px; /* 圓角 */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        if st.button('開始預測'):
            # 呼叫預測函數
            prediction_dict = predict(image_path)
            st.session_state['prediction_dict'] = prediction_dict

with col2:
    if 'prediction_dict' in st.session_state:
        st.markdown(
    '''
    <div class="result-title" style="text-align: center; font-size: 30px; font-weight: bold; margin-top: -40 px;">
        預測結果
    </div>
    ''', 
    unsafe_allow_html=True
)
        max_prob = max(st.session_state['prediction_dict'].values())
        st.markdown('<div class="grid-container">', unsafe_allow_html=True)

        for disease, prob in st.session_state['prediction_dict'].items():
            display_name = disease_display_names.get(disease, disease)  # 若沒定義對應則顯示原縮寫
            box_class = 'highlight-box' if prob == max_prob else 'prediction-box'
            st.markdown(f'<div class="{box_class}">{display_name}: {prob*100:.2f}%</div>', unsafe_allow_html=True)


        st.markdown('</div>', unsafe_allow_html=True)

# 網頁底部的版權標示
st.markdown(
    '''
    <div class="footer">
        © 2025 CMU Artificial Intelligence & Bioimaging Lab, An Nan Hospital. All rights reserved.<br>
        Developed by: Chen-Hao Peng & Da-Chuan Cheng & Tzu-Kun Lo & Jr-Rong Wang
    </div>
    ''', 
    unsafe_allow_html=True
)

# 在 sidebar 的最底部加上一條水平線作區隔，再放上 Lab Logo
st.sidebar.markdown("---")  # 水平線，可依需求保留或移除
st.sidebar.markdown(
    f"""
    <div style="text-align: center; margin-top: 10px;">
        <img src="data:image/png;base64,{lab_logo}" alt="Lab Logo" style="max-width: 150px;">
    </div>
    """,
    unsafe_allow_html=True
)
