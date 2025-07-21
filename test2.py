import os
import io
import cv2
import numpy as np
import logging
import json
import tempfile
import requests
from datetime import datetime
from flask import Flask, request, abort, jsonify
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

# Import configuration
try:
    from config import Config
    Config.validate()
except ImportError:
    print("❌ config.py not found. Please run setup.py first.")
    exit(1)
except ValueError as e:
    print(f"❌ Configuration Error: {e}")
    print("Please check your .env file and ensure all required values are set.")
    exit(1)

# === Initialize Flask App ===
app = Flask(__name__)

# === Initialize LINE Bot API ===
line_bot_api = LineBotApi(Config.LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(Config.LINE_CHANNEL_SECRET)

# === Setup Logging ===
def setup_logging():
    """ตั้งค่า logging system"""
    if not os.path.exists(Config.LOG_DIR):
        os.makedirs(Config.LOG_DIR)
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'{Config.LOG_DIR}/line_bot_{timestamp}.log'
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
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
    face_app = FaceAnalysis(name=Config.FACE_MODEL_NAME, providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=Config.FACE_DETECTION_SIZE)
    logger.info("✅ โหลดโมเดล InsightFace สำเร็จ")
except Exception as e:
    logger.error(f"❌ โหลดโมเดลไม่สำเร็จ: {str(e)}")
    raise

# === Google Drive Setup ===
def authenticate_google_drive():
    """ยืนยันตัวตนกับ Google Drive"""
    creds = None
    
    # โหลด token ที่บันทึกไว้
    if os.path.exists(Config.GOOGLE_TOKEN_FILE):
        with open(Config.GOOGLE_TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    
    # ถ้าไม่มี credentials ที่ใช้ได้
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                Config.GOOGLE_CREDENTIALS_FILE, Config.GOOGLE_SCOPES)
            creds = flow.run_local_server(port=0)
        
        # บันทึก credentials
        with open(Config.GOOGLE_TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)

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
        drive_images = get_images_from_drive_folder(service, Config.GOOGLE_DRIVE_FOLDER_ID)
        
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
        
        if best_score >= Config.COSINE_SIM_THRESHOLD:
            logger.info(f"✅ พบภาพที่ตรงกัน: {best_image['name']} ({best_score:.4f})")
            return best_image, f"พบภาพที่ตรงกัน: {best_image['name']} (ความคล้าย: {best_score:.2%})"
        else:
            logger.info(f"❌ ไม่พบภาพที่ตรงกัน (คะแนนสูงสุด: {best_score:.4f})")
            return None, f"ไม่พบภาพที่ตรงกัน (ความคล้ายสูงสุด: {best_score:.2%})"
            
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการค้นหา: {str(e)}")
        return None, f"เกิดข้อผิดพลาดในการค้นหา: {str(e)}"

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
        if width > Config.MAX_IMAGE_SIZE or height > Config.MAX_IMAGE_SIZE:
            scale = min(Config.MAX_IMAGE_SIZE/width, Config.MAX_IMAGE_SIZE/height)
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
        
        logger.info(f"✅ ตรวจพบใบหน้าเรียบร้อย")
        return main_face.embedding
        
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")
        return None