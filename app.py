import os
import io
import cv2
import numpy as np
import logging
import json
import tempfile
import requests
from datetime import datetime
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage, 
    ImageSendMessage, FlexSendMessage, BubbleContainer, 
    BoxComponent, TextComponent, ImageComponent, ButtonComponent,
    PostbackAction, MessageAction
)
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from insightface.app import FaceAnalysis
from tqdm import tqdm

# === Configuration ===
LINE_CHANNEL_ACCESS_TOKEN = 'oA4uOEDW7hKpmkO94X0GiSPZ5Kej0QM8vbkCglUC0HLRG/OF9j35XOGTdx0B+bH2rC+70ajDn5jSrj2SBixirpuifvbIBl0Zz48AUMdIWrOPS90B8/IhIF7hs8L5svaRTA82nCGQhfCx2j09JNXuzwdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = '603e1fd9cb40d7fd19de468c824cacd5'
GOOGLE_DRIVE_FOLDER_ID = '1BNmuKRL1vQE2czc9snVg48-S7O6fAjSd'
COSINE_SIM_THRESHOLD = 0.4

# === Initialize Flask App ===
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return 'OK', 200

# === Initialize LINE Bot API ===
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# === Setup Logging ===
def setup_logging():
    """ตั้งค่า logging system"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/line_bot_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 LINE Bot เริ่มทำงาน - Log: {log_filename}")
    return logger

logger = setup_logging()

# === Initialize InsightFace ===
logger.info("📥 กำลังโหลดโมเดล InsightFace...")
try:
    face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("✅ โหลดโมเดล InsightFace สำเร็จ")
except Exception as e:
    logger.error(f"❌ โหลดโมเดลไม่สำเร็จ: {str(e)}")
    raise

# === Google Drive Setup ===
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_google_drive():
    """ยืนยันตัวตนกับ Google Drive"""
    creds = None
    
    # โหลด token ที่บันทึกไว้
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # ถ้าไม่มี credentials ที่ใช้ได้
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # บันทึก credentials
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)

# === Face Recognition Functions ===
def cosine_similarity(a, b):
    """คำนวณ cosine similarity"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a, b) / (norm_a * norm_b)

def get_face_embedding_from_bytes(image_bytes):
    """ดึง embedding จาก image bytes"""
    try:
        # แปลง bytes เป็น numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("❌ ไม่สามารถ decode ภาพได้")
            return None
        
        # ปรับขนาดภาพถ้าจำเป็น
        height, width = img.shape[:2]
        if width > 1024 or height > 1024:
            scale = min(1024/width, 1024/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # ตรวจจับใบหน้า
        faces = face_app.get(img)
        if not faces:
            logger.warning("⚠️ ไม่พบใบหน้าในภาพ")
            return None
        
        # เลือกใบหน้าที่ใหญ่ที่สุด
        main_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        logger.info(f"✅ ตรวจจับใบหน้าสำเร็จ - จำนวน: {len(faces)}")
        return main_face.embedding
        
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")
        return None

def download_image_from_drive(service, file_id):
    """ดาวน์โหลดภาพจาก Google Drive"""
    try:
        request = service.files().get_media(fileId=file_id)
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        file_io.seek(0)
        return file_io.read()
        
    except Exception as e:
        logger.error(f"❌ ไม่สามารถดาวน์โหลดไฟล์ {file_id}: {str(e)}")
        return None

def get_images_from_drive_folder(service, folder_id):
    """ดึงรายการภาพจาก Google Drive folder"""
    try:
        # ค้นหาไฟล์ในโฟลเดอร์
        query = f"'{folder_id}' in parents and mimeType contains 'image/'"
        results = service.files().list(
            q=query,
            pageSize=1000,
            fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
        
        items = results.get('files', [])
        logger.info(f"📁 พบภาพใน Google Drive: {len(items)} ไฟล์")
        
        return items
        
    except Exception as e:
        logger.error(f"❌ ไม่สามารถเข้าถึง Google Drive folder: {str(e)}")
        return []

def find_similar_face_in_drive(user_image_bytes, user_id):
    """ค้นหาใบหน้าที่คล้ายกันใน Google Drive"""
    logger.info(f"🔍 เริ่มค้นหาใบหน้าสำหรับ user: {user_id}")
    
    # ดึง embedding ของภาพผู้ใช้
    user_embedding = get_face_embedding_from_bytes(user_image_bytes)
    if user_embedding is None:
        return None, "ไม่พบใบหน้าในภาพที่ส่งมา กรุณาส่งภาพที่มีใบหน้าชัดเจน"
    
    try:
        # เชื่อมต่อ Google Drive
        service = authenticate_google_drive()
        drive_images = get_images_from_drive_folder(service, GOOGLE_DRIVE_FOLDER_ID)
        
        if not drive_images:
            return None, "ไม่พบภาพใน Google Drive"
        
        best_score = -1.0
        best_image = None
        processed_count = 0
        
        logger.info(f"📊 กำลังเปรียบเทียบกับ {len(drive_images)} ภาพ")
        
        # เปรียบเทียบกับแต่ละภาพใน Drive
        for image_info in drive_images:
            try:
                # ดาวน์โหลดภาพ
                image_bytes = download_image_from_drive(service, image_info['id'])
                if image_bytes is None:
                    continue
                
                # ดึง embedding
                target_embedding = get_face_embedding_from_bytes(image_bytes)
                if target_embedding is None:
                    continue
                
                # คำนวณความคล้าย
                score = cosine_similarity(user_embedding, target_embedding)
                processed_count += 1
                
                logger.debug(f"📊 {image_info['name']}: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_image = image_info
                    
            except Exception as e:
                logger.error(f"❌ เกิดข้อผิดพลาดขณะประมวลผล {image_info['name']}: {str(e)}")
                continue
        
        logger.info(f"📈 ประมวลผลเสร็จสิ้น: {processed_count}/{len(drive_images)} ภาพ")
        logger.info(f"🏆 คะแนนสูงสุด: {best_score:.4f}")
        
        if best_score >= COSINE_SIM_THRESHOLD:
            logger.info(f"✅ พบภาพที่ตรงกัน: {best_image['name']} ({best_score:.4f})")
            return best_image, f"พบภาพที่ตรงกัน: {best_image['name']} (ความคล้าย: {best_score:.2%})"
        else:
            logger.info(f"❌ ไม่พบภาพที่ตรงกัน (คะแนนสูงสุด: {best_score:.4f})")
            return None, f"ไม่พบภาพที่ตรงกัน (ความคล้ายสูงสุด: {best_score:.2%})"
            
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการค้นหา: {str(e)}")
        return None, f"เกิดข้อผิดพลาดในการค้นหา: {str(e)}"

# === LINE Bot Functions ===
def create_result_flex_message(result_image, message, similarity_score=None):
    """สร้าง Flex Message สำหรับแสดงผลลัพธ์"""
    if result_image:
        # สร้าง Google Drive view link
        drive_link = f"https://drive.google.com/file/d/{result_image['id']}/view"
        
        bubble = BubbleContainer(
            direction='ltr',
            body=BoxComponent(
                layout='vertical',
                contents=[
                    TextComponent(
                        text='🎉 พบภาพที่ตรงกัน!',
                        weight='bold',
                        size='xl',
                        color='#1DB446'
                    ),
                    TextComponent(
                        text=f"📁 {result_image['name']}",
                        size='md',
                        color='#666666',
                        margin='md'
                    ),
                    TextComponent(
                        text=message,
                        size='sm',
                        color='#999999',
                        margin='sm'
                    )
                ]
            ),
            footer=BoxComponent(
                layout='vertical',
                contents=[
                    ButtonComponent(
                        action=MessageAction(
                            label='ดูภาพใน Google Drive',
                            text=f'ดูภาพ: {result_image["name"]}'
                        ),
                        color='#1DB446'
                    )
                ]
            )
        )
    else:
        bubble = BubbleContainer(
            direction='ltr',
            body=BoxComponent(
                layout='vertical',
                contents=[
                    TextComponent(
                        text='😔 ไม่พบภาพที่ตรงกัน',
                        weight='bold',
                        size='xl',
                        color='#FF5551'
                    ),
                    TextComponent(
                        text=message,
                        size='md',
                        color='#666666',
                        margin='md'
                    ),
                    TextComponent(
                        text='💡 ลองส่งภาพอื่นที่มีใบหน้าชัดเจนกว่า',
                        size='sm',
                        color='#999999',
                        margin='md'
                    )
                ]
            )
        )
    
    return FlexSendMessage(alt_text='ผลลัพธ์การค้นหา', contents=bubble)

@app.route("/callback", methods=['POST'])
def callback():
    """รับ webhook จาก LINE"""
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("❌ Invalid signature")
        abort(400)
    
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    """จัดการข้อความธรรมดา"""
    user_id = event.source.user_id
    text = event.message.text
    
    logger.info(f"📝 ข้อความจาก {user_id}: {text}")
    
    if text.lower() in ['help', 'ช่วยเหลือ', 'วิธีใช้']:
        help_message = """
🤖 วิธีใช้งาน Face Recognition Bot

1. ส่งภาพที่มีใบหน้าของคุณ
2. ระบบจะค้นหาภาพที่คล้ายกันใน Google Drive
3. รับผลลัพธ์และลิงค์ไปยังภาพที่ตรงกัน

⚠️ ข้อควรระวัง:
- ใช้ภาพที่มีใบหน้าชัดเจน
- หลีกเลี่ยงภาพที่มีแสงแรงหรือเงา
- ภาพควรมีขนาดไม่เกิน 10MB
        """
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=help_message)
        )
    else:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="กรุณาส่งภาพที่มีใบหน้าของคุณ หรือพิมพ์ 'help' เพื่อดูวิธีใช้งาน")
        )

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    """จัดการภาพที่ส่งมา"""
    user_id = event.source.user_id
    message_id = event.message.id
    
    logger.info(f"📸 รับภาพจาก {user_id}, message_id: {message_id}")
    
    try:
        # ส่งข้อความแจ้งว่ากำลังประมวลผล
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="🔍 กำลังประมวลผลภาพ กรุณารอสักครู่...")
        )
        
        # ดาวน์โหลดภาพจาก LINE
        message_content = line_bot_api.get_message_content(message_id)
        image_bytes = b''
        for chunk in message_content.iter_content():
            image_bytes += chunk
        
        logger.info(f"📥 ดาวน์โหลดภาพสำเร็จ ({len(image_bytes)} bytes)")
        
        # ค้นหาใบหน้าที่คล้ายกัน
        result_image, message = find_similar_face_in_drive(image_bytes, user_id)
        
        # ส่งผลลัพธ์กลับ
        flex_message = create_result_flex_message(result_image, message)
        line_bot_api.push_message(user_id, flex_message)
        
        logger.info(f"✅ ส่งผลลัพธ์ให้ {user_id} เรียบร้อย")
        
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")
        )

if __name__ == "__main__":
    logger.info("🌟 เริ่มต้น LINE Bot Server")
    app.run(host='0.0.0.0', port=5000, debug=False)