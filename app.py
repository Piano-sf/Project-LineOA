import os
import io
import cv2
import numpy as np
import logging
import json
import tempfile
import base64
import requests
import time
from urllib.parse import quote
from datetime import datetime
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage, 
    ImageSendMessage, FlexSendMessage, BubbleContainer, 
    BoxComponent, TextComponent, ImageComponent, ButtonComponent,
    PostbackAction, MessageAction, URIAction, MessageAction,
    CarouselContainer
)
# from google.oauth2.credentials import Credentials
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaIoBaseDownload
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# import pickle
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from insightface.app import FaceAnalysis
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration with environment variables
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
GOOGLE_DRIVE_FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID', '1BNmuKRL1vQE2czc9snVg48-S7O6fAjSd')
COSINE_SIM_THRESHOLD = float(os.getenv('COSINE_SIM_THRESHOLD', '0.35'))

# Validate required environment variables
required_vars = ['LINE_CHANNEL_ACCESS_TOKEN', 'LINE_CHANNEL_SECRET']
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Validate required files
required_files = ['service-account-key.json']
missing_files = [file for file in required_files if not os.path.exists(file)]

if missing_files:
    raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

print("✅ Configuration validated successfully")

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
SCOPES = ['https://www.googleapis.com/auth/drive']

# แทนที่ฟังก์ชัน authenticate_google_drive() เดิมด้วยอันนี้
def authenticate_google_drive():
    """ยืนยันตัวตนกับ Google Drive ด้วย Service Account"""
    try:
        # ตรวจสอบว่ามีไฟล์ service account key
        if not os.path.exists('service-account-key.json'):
            raise FileNotFoundError("ไม่พบไฟล์ service-account-key.json")
        
        logger.info("🔑 กำลังโหลด service-account-key.json...")
        
        # สร้าง credentials จาก service account key
        credentials = service_account.Credentials.from_service_account_file(
            'service-account-key.json',
            scopes=SCOPES
        )
        
        # สร้าง Google Drive service
        service = build('drive', 'v3', credentials=credentials)
        
        # ทดสอบการเชื่อมต่อ
        test_result = service.files().list(pageSize=1).execute()
        logger.info("✅ เชื่อมต่อ Google Drive API ด้วย Service Account สำเร็จ")
        
        return service
        
    except FileNotFoundError as e:
        logger.error(f"❌ {str(e)}")
        logger.error("💡 วิธีแก้ไข:")
        logger.error("1. สร้าง Service Account ใน Google Cloud Console")
        logger.error("2. ดาวน์โหลด JSON key file")
        logger.error("3. เปลี่ยนชื่อเป็น 'service-account-key.json'")
        logger.error("4. วางไฟล์ในโฟลเดอร์เดียวกับ app.py")
        raise
        
    except Exception as e:
        logger.error(f"❌ ไม่สามารถ authenticate ด้วย Service Account ได้: {str(e)}")
        logger.error("💡 ตรวจสอบ:")
        logger.error("1. ไฟล์ service-account-key.json ถูกต้องหรือไม่")
        logger.error("2. Google Drive API เปิดใช้งานแล้วหรือไม่")
        logger.error("3. Service Account มีสิทธิ์เข้าถึงโฟลเดอร์หรือไม่")
        raise

def test_google_drive_connection():
    """ทดสอบการเชื่อมต่อ Google Drive"""
    try:
        service = authenticate_google_drive()
        
        # ทดสอบเข้าถึงโฟลเดอร์
        logger.info(f"🔍 ทดสอบการเข้าถึงโฟลเดอร์: {GOOGLE_DRIVE_FOLDER_ID}")
        
        query = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and mimeType contains 'image/'"
        results = service.files().list(
            q=query,
            pageSize=5,
            fields="files(id, name, mimeType, size)"
        ).execute()
        
        files = results.get('files', [])
        logger.info(f"✅ พบ {len(files)} ไฟล์รูปภาพในโฟลเดอร์")
        
        if files:
            logger.info("📁 ตัวอย่างไฟล์:")
            for i, file in enumerate(files[:3], 1):
                size_mb = int(file.get('size', 0)) / (1024*1024) if file.get('size') else 0
                logger.info(f"  {i}. {file['name']} ({size_mb:.1f} MB)")
        else:
            logger.warning("⚠️ ไม่พบไฟล์รูปภาพในโฟลเดอร์")
            logger.warning("💡 ตรวจสอบ:")
            logger.warning("1. GOOGLE_DRIVE_FOLDER_ID ถูกต้องหรือไม่")
            logger.warning("2. Service Account ได้รับสิทธิ์เข้าถึงโฟลเดอร์หรือไม่")
            logger.warning("3. โฟลเดอร์มีไฟล์รูปภาพหรือไม่")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ทดสอบการเชื่อมต่อไม่สำเร็จ: {str(e)}")
        return False

# ฟังก์ชันตรวจสอบ Service Account Email
def show_service_account_info():
    """แสดงข้อมูล Service Account สำหรับการแชร์โฟลเดอร์"""
    try:
        if os.path.exists('service-account-key.json'):
            with open('service-account-key.json', 'r') as f:
                key_data = json.load(f)
            
            client_email = key_data.get('client_email', 'ไม่ทราบ')
            project_id = key_data.get('project_id', 'ไม่ทราบ')
            
            logger.info("📧 ข้อมูล Service Account:")
            logger.info(f"   Email: {client_email}")
            logger.info(f"   Project: {project_id}")
            logger.info("💡 แชร์โฟลเดอร์ Google Drive ด้วย email นี้")
            
            return client_email
        else:
            logger.warning("⚠️ ไม่พบไฟล์ service-account-key.json")
            return None
            
    except Exception as e:
        logger.error(f"❌ ไม่สามารถอ่านข้อมูล Service Account ได้: {str(e)}")
        return None

# === ปรับปรุง Face Recognition Functions ===

def enhance_image_quality(img):
    """ปรับปรุงคุณภาพภาพก่อนประมวลผล"""
    try:
        # แปลงเป็น LAB color space เพื่อปรับ lighting
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # ใช้ CLAHE สำหรับปรับ contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # รวมกลับเป็น BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # ลด noise
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        logger.debug("✅ ปรับปรุงคุณภาพภาพสำเร็จ")
        return enhanced
        
    except Exception as e:
        logger.warning(f"⚠️ ไม่สามารถปรับปรุงคุณภาพภาพได้: {str(e)}")
        return img

def normalize_face_orientation(img, face):
    """ปรับ orientation ของใบหน้าให้ตรง"""
    try:
        if not hasattr(face, 'kps') or len(face.kps) < 2:
            return img
        
        left_eye = face.kps[0]
        right_eye = face.kps[1]
        
        # คำนวณมุมการหมุน
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # หมุนเฉพาะเมื่อมุมเอียงมาก
        if abs(angle) > 5:
            center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
            logger.debug(f"✅ ปรับ orientation สำเร็จ (มุม: {angle:.2f}°)")
            return rotated
        
        return img
        
    except Exception as e:
        logger.warning(f"⚠️ ไม่สามารถปรับ orientation ได้: {str(e)}")
        return img

def calculate_face_quality_score(face, img_shape):
    """คำนวณคะแนนคุณภาพของใบหน้า"""
    try:
        bbox = face.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # คะแนนขนาด (ใหญ่กว่า = ดีกว่า)
        size_score = min((width * height) / (img_shape[0] * img_shape[1]) * 10, 1.0)
        
        # คะแนนตำแหน่ง (กลางภาพ = ดีกว่า)
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        img_center_x = img_shape[1] / 2
        img_center_y = img_shape[0] / 2
        
        distance_from_center = np.sqrt(
            (center_x - img_center_x)**2 + (center_y - img_center_y)**2
        )
        max_distance = np.sqrt(img_center_x**2 + img_center_y**2)
        position_score = 1 - (distance_from_center / max_distance)
        
        # คะแนนความชัด (det_score จาก InsightFace)
        detection_score = face.det_score if hasattr(face, 'det_score') else 0.5
        
        # คะแนนอัตราส่วนใบหน้า
        aspect_ratio = width / height if height > 0 else 0
        aspect_score = 1.0 if 0.7 <= aspect_ratio <= 1.3 else 0.5
        
        # รวมคะแนน
        total_score = (size_score * 0.3) + (position_score * 0.2) + (detection_score * 0.3) + (aspect_score * 0.2)
        
        return min(total_score, 1.0)
        
    except Exception as e:
        logger.warning(f"⚠️ ไม่สามารถคำนวณคะแนนคุณภาพได้: {str(e)}")
        return 0.5

def select_best_face(faces, img_shape):
    """เลือกใบหน้าที่ดีที่สุดตามคะแนนคุณภาพ"""
    if not faces:
        return None
    
    best_face = None
    best_score = -1.0
    
    for face in faces:
        score = calculate_face_quality_score(face, img_shape)
        logger.debug(f"📊 Face quality score: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_face = face
    
    logger.info(f"✅ เลือกใบหน้าที่ดีที่สุด (คะแนน: {best_score:.3f})")
    return best_face

def advanced_similarity_calculation(embedding1, embedding2, method='weighted'):
    """คำนวณความคล้ายแบบปรับปรุง"""
    try:
        if method == 'cosine':
            return cosine_similarity(embedding1, embedding2)
        
        elif method == 'euclidean':
            distance = np.linalg.norm(embedding1 - embedding2)
            similarity = 1 / (1 + distance)
            return similarity
        
        elif method == 'weighted':
            # รวมหลายวิธี
            cosine_sim = cosine_similarity(embedding1, embedding2)
            euclidean_distance = np.linalg.norm(embedding1 - embedding2)
            euclidean_sim = 1 / (1 + euclidean_distance)
            
            # ถ่วงน้ำหนัก
            weighted_sim = (cosine_sim * 0.7) + (euclidean_sim * 0.3)
            return weighted_sim
        
        elif method == 'manhattan':
            distance = np.sum(np.abs(embedding1 - embedding2))
            similarity = 1 / (1 + distance * 0.1)
            return similarity
        
        else:
            return cosine_similarity(embedding1, embedding2)
            
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการคำนวณความคล้าย: {str(e)}")
        return 0.0

def adaptive_threshold(face_quality_score, base_threshold=None):
    """ปรับ threshold ตามคุณภาพของใบหน้า"""
    if base_threshold is None:
        base_threshold = COSINE_SIM_THRESHOLD
    
    # ปรับตามคุณภาพใบหน้า
    if face_quality_score > 0.8:
        return base_threshold + 0.05  # เข้มงวดมากขึ้น
    elif face_quality_score > 0.6:
        return base_threshold
    elif face_quality_score > 0.35:
        return base_threshold - 0.05  # ผ่อนปรนเล็กน้อย
    else:
        return base_threshold - 0.1   # ผ่อนปรนมากขึ้น

def calculate_confidence_score(similarity, quality_score, face_size):
    """คำนวณความมั่นใจในผลลัพธ์"""
    try:
        # Base confidence จาก similarity
        base_confidence = similarity
        
        # ปรับตามคุณภาพใบหน้า
        quality_factor = min(quality_score * 1.2, 1.0)
        
        # ปรับตามขนาดใบหน้า (normalize ด้วย 10000 pixels)
        size_factor = min(face_size / 10000, 1.0)
        
        # คำนวณ confidence รวม
        confidence = base_confidence * quality_factor * size_factor
        
        return min(confidence, 1.0)
        
    except Exception as e:
        logger.warning(f"⚠️ ไม่สามารถคำนวณ confidence ได้: {str(e)}")
        return similarity

def cosine_similarity(a, b):
    """คำนวณ cosine similarity"""
    try:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
        
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการคำนวณ cosine similarity: {str(e)}")
        return 0.0

def get_face_embedding_from_bytes(image_bytes, enhanced=True):
    """ดึง embedding จาก image bytes แบบปรับปรุง"""
    try:
        # แปลง bytes เป็น numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("❌ ไม่สามารถ decode ภาพได้")
            return None, None, None
        
        # ปรับขนาดภาพถ้าจำเป็น
        height, width = img.shape[:2]
        if width > 1024 or height > 1024:
            scale = min(1024/width, 1024/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            logger.debug(f"📏 ปรับขนาดภาพ: {width}x{height} -> {new_width}x{new_height}")
        
        # ปรับปรุงคุณภาพภาพ
        if enhanced:
            img = enhance_image_quality(img)
        
        # ตรวจจับใบหน้า
        faces = face_app.get(img)
        if not faces:
            logger.warning("⚠️ ไม่พบใบหน้าในภาพ")
            return None, None, None
        
        # เลือกใบหน้าที่ดีที่สุด
        best_face = select_best_face(faces, img.shape)
        if best_face is None:
            logger.warning("⚠️ ไม่สามารถเลือกใบหน้าได้")
            return None, None, None
        
        # ปรับ orientation
        if enhanced:
            img = normalize_face_orientation(img, best_face)
            # ตรวจจับใบหน้าใหม่หลังปรับ orientation
            normalized_faces = face_app.get(img)
            if normalized_faces:
                best_face = select_best_face(normalized_faces, img.shape)
        
        # คำนวณคะแนนคุณภาพ
        quality_score = calculate_face_quality_score(best_face, img.shape)
        
        # คำนวณขนาดใบหน้า
        bbox = best_face.bbox
        face_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        logger.info(f"✅ ตรวจจับใบหน้าสำเร็จ - จำนวน: {len(faces)}, คุณภาพ: {quality_score:.3f}")
        
        return best_face.embedding, quality_score, face_size
        
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")
        return None, None, None

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

def log_matching_metrics(matches, processing_time, user_id):
    """บันทึก metrics สำหรับการปรับปรุงระบบ"""
    try:
        metrics = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'processing_time': float(processing_time),
            'total_matches': int(len(matches)),
            'high_confidence_matches': int(len([m for m in matches if float(m.get('similarity', 0)) > 0.7])),
            'average_similarity': float(np.mean([m.get('similarity', 0) for m in matches])) if matches else 0,
            'max_similarity': float(max([m.get('similarity', 0) for m in matches])) if matches else 0
        }
        
        logger.info(f"📊 Matching Metrics: {json.dumps(metrics, indent=2)}")
        return metrics
        
    except Exception as e:
        logger.error(f"❌ ไม่สามารถบันทึก metrics ได้: {str(e)}")
        return {}

def find_similar_face_in_drive(user_image_bytes, user_id, max_results=5):
    """ค้นหาใบหน้าที่คล้ายกันใน Google Drive แล้วส่งกลับสูงสุด max_results ภาพ"""
    start_time = time.time()
    logger.info(f"🔍 เริ่มค้นหาใบหน้าสำหรับ user: {user_id}")

    # 1️⃣ ดึงใบหน้าผู้ใช้ (รองรับหลายหน้า)
    user_faces = get_all_face_embeddings_from_bytes(user_image_bytes, enhanced=True)
    if not user_faces:
        return None, "ไม่พบใบหน้าในภาพที่ส่งมา กรุณาส่งภาพที่มีใบหน้าชัดเจน"

    logger.info(f"👥 ตรวจพบใบหน้าทั้งหมด: {len(user_faces)} ใบหน้า")

    try:
        # 2️⃣ โหลดภาพจาก Google Drive
        service = authenticate_google_drive()
        drive_images = get_images_from_drive_folder(service, GOOGLE_DRIVE_FOLDER_ID)
        if not drive_images:
            return None, "ไม่พบภาพใน Google Drive"

        matched_images = {}  # 📁 เก็บเฉพาะภาพที่ match อย่างน้อย 1 หน้า
        processed_count = 0
        best_score = -1.0

        logger.info(f"📊 กำลังเปรียบเทียบกับ {len(drive_images)} ภาพ")

        # 3️⃣ วนทุกใบหน้าผู้ใช้
        for user_idx, user_face in enumerate(user_faces, start=1):
            user_embedding = user_face["embedding"]
            quality_score = user_face["quality_score"]
            face_size = user_face["face_size"]

            dynamic_threshold = adaptive_threshold(quality_score)
            logger.info(f"🎯 Face #{user_idx}: Threshold = {dynamic_threshold:.3f}")

            for image_info in drive_images:
                try:
                    file_id = image_info["id"]

                    if file_id in matched_images:
                        continue  # skip already matched

                    image_bytes = download_image_from_drive(service, file_id)
                    if image_bytes is None:
                        continue

                    target_faces = get_all_face_embeddings_from_bytes(image_bytes, enhanced=True)
                    if not target_faces:
                        continue

                    for target in target_faces:
                        target_embedding = target["embedding"]
                        target_quality = target["quality_score"]
                        target_face_size = target["face_size"]

                        # 💡 กรองใบหน้าคุณภาพต่ำ
                        if target_quality < 0.4 or target_face_size < 1500:
                            continue

                        similarity = advanced_similarity_calculation(
                            user_embedding, target_embedding, method='weighted'
                        )

                        confidence = calculate_confidence_score(
                            similarity,
                            min(quality_score, target_quality),
                            min(face_size, target_face_size)
                        )

                        processed_count += 1

                        if similarity >= dynamic_threshold:
                            matched_images[file_id] = {
                                'image_info': image_info,
                                'similarity': similarity,
                                'confidence': confidence
                            }

                            if similarity > best_score:
                                best_score = similarity

                            break  # ✅ เจอแมทช์แล้วในภาพนี้ ไม่ต้องวนใบหน้าอื่น

                except Exception as e:
                    logger.error(f"❌ Error processing {image_info['name']}: {str(e)}")
                    continue

        # 4️⃣ สรุปผล
        processing_time = time.time() - start_time
        logger.info(f"📈 ประมวลผลเสร็จสิ้น: {processed_count} ใบหน้าใน {processing_time:.2f} วินาที")

        all_matches = list(matched_images.values())
        all_matches.sort(key=lambda x: x["confidence"], reverse=True)
        top_matches = all_matches[:max_results]

        logger.info(f"🏆 พบภาพที่ตรงกันทั้งหมด: {len(all_matches)} / ส่งกลับ: {len(top_matches)}")

        if top_matches:
            return top_matches, f"พบภาพที่ตรงกัน {len(top_matches)} ภาพ"
        else:
            return [], f"ไม่พบภาพที่ตรงกัน\nความคล้ายสูงสุด: {best_score:.2%}"

    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการค้นหา: {str(e)}")
        return None, f"เกิดข้อผิดพลาด: {str(e)}"
    
def get_all_face_embeddings_from_bytes(image_bytes, enhanced=True):
    """ดึง embeddings ของใบหน้าทุกใบในภาพ"""
    try:
        # แปลง bytes เป็น numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("❌ ไม่สามารถ decode ภาพได้")
            return []

        # ปรับขนาดภาพถ้าจำเป็น (ช่วยให้ประมวลผลเร็วขึ้น)
        height, width = img.shape[:2]
        if width > 1024 or height > 1024:
            scale = min(1024/width, 1024/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            logger.debug(f"📏 ปรับขนาดภาพ: {width}x{height} -> {new_width}x{new_height}")

        # ปรับปรุงคุณภาพภาพ
        if enhanced:
            img = enhance_image_quality(img)

        # 🔍 ตรวจจับใบหน้า
        faces = face_app.get(img)
        if not faces:
            logger.warning("⚠️ ไม่พบใบหน้าในภาพ")
            return []

        results = []
        for face in faces:
            # ปรับ orientation เฉพาะใบหน้านี้ (optional)
            if enhanced:
                img = normalize_face_orientation(img, face)

            # คำนวณคุณภาพของใบหน้า
            quality_score = calculate_face_quality_score(face, img.shape)

            # คำนวณขนาดใบหน้า
            bbox = face.bbox
            face_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            results.append({
                "embedding": face.embedding,
                "quality_score": quality_score,
                "face_size": face_size
            })

            logger.info(f"✅ ตรวจจับใบหน้าสำเร็จ (คุณภาพ: {quality_score:.3f})")

        logger.info(f"👥 พบใบหน้าทั้งหมด {len(results)} ใบหน้า")
        return results

    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลภาพ (multi-face): {str(e)}")
        return []

# === Flex Message Creation ===
def create_result_flex_message(result_image, message, similarity_score=None, image_url=None):
    """สร้าง Flex Message พร้อมปุ่มเปิดภาพ"""
    if result_image:
        # ถ้าไม่มี image_url ให้ใช้ Google Drive view link แทน
        if not image_url:
            image_url = f"https://drive.google.com/file/d/{result_image['id']}/view"

        contents = [
            TextComponent(
                text='🎉 พบภาพที่ตรงกัน!',
                weight='bold',
                size='xl',
                color='#1DB446'
            )
        ]

        # แสดงภาพใน Flex ถ้ามี
        if image_url:
            contents.append(
                ImageComponent(
                    url=image_url,
                    size='full',
                    aspect_ratio='1:1',
                    aspect_mode='cover',
                    margin='md'
                )
            )

        contents.extend([
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
                margin='sm',
                wrap=True
            )
        ])

        # ถ้าได้ลิงก์ ใช้ URIAction; ถ้าไม่ได้ ใช้ MessageAction
        action = URIAction(label='เปิดภาพเต็ม', uri=image_url) if image_url else MessageAction(label='ดูรายละเอียด', text=f'ดูภาพ: {result_image["name"]}')

        bubble = BubbleContainer(
            direction='ltr',
            body=BoxComponent(layout='vertical', contents=contents),
            footer=BoxComponent(
                layout='vertical',
                contents=[
                    ButtonComponent(
                        action=action,
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
                        margin='md',
                        wrap=True
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

# === LINE Bot Functions ===
def create_public_drive_link(file_id):
    """สร้าง public link สำหรับ Google Drive file"""
    try:
        service = authenticate_google_drive()
        
        # ทำให้ไฟล์เป็น public (อ่านได้โดยทุกคน)
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        
        service.permissions().create(
            fileId=file_id,
            body=permission
        ).execute()
        
        # สร้าง direct download link (ใช้สำหรับ LINE)
        direct_link = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        logger.info(f"✅ สร้าง public link สำเร็จ: {direct_link}")
        return direct_link
        
    except Exception as e:
        logger.error(f"❌ ไม่สามารถสร้าง public link ได้: {str(e)}")
        return None
    
def create_carousel_message(matches):
    bubbles = []
    for m in matches[:5]:  # each m is a dict
        image_info = m['image_info']
        similarity = float(m['similarity'])
        file_id = image_info['id']
        image_url = create_public_drive_link(file_id) or f"https://drive.google.com/file/d/{file_id}/view"

        bubble = BubbleContainer(
            direction='ltr',
            body=BoxComponent(
                layout='vertical',
                contents=[
                    ImageComponent(
                        url=image_url,
                        size='full',
                        aspect_mode='cover',
                        aspect_ratio='1:1'
                    ),
                    TextComponent(
                        text=image_info['name'],
                        weight='bold',
                        size='md',
                        wrap=True,
                        margin='md'
                    ),
                    TextComponent(
                        text=f"ความคล้าย: {similarity:.2%}",
                        size='sm',
                        color='#666666',
                        margin='sm'
                    )
                ]
            ),
            footer=BoxComponent(
                layout='vertical',
                contents=[
                    ButtonComponent(
                        action=URIAction(
                            label='เปิดภาพเต็ม',
                            uri=image_url
                        ),
                        style='primary',
                        color='#1DB446'
                    )
                ]
            )
        )
        bubbles.append(bubble)

    carousel = CarouselContainer(contents=bubbles)
    return FlexSendMessage(alt_text='ผลลัพธ์การค้นหา', contents=carousel)

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    user_id = event.source.user_id
    message_id = event.message.id
    logger.info(f"📸 รับภาพจาก {user_id}, message_id: {message_id}")
    
    try:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="🔍 กำลังประมวลผลภาพ กรุณารอสักครู่...")
        )

        # ดาวน์โหลดภาพจาก LINE
        message_content = line_bot_api.get_message_content(message_id)
        image_bytes = b''.join(chunk for chunk in message_content.iter_content())
        logger.info(f"📥 ดาวน์โหลดภาพสำเร็จ ({len(image_bytes)} bytes)")

        # ค้นหาใบหน้าที่คล้ายกัน
        matches, message = find_similar_face_in_drive(image_bytes, user_id, max_results=50)

        if matches:
            def chunked(lst, n):
                for i in range(0, len(lst), n):
                    yield lst[i:i + n]
                    
            if matches:
                for m in matches:
                    image_info = m['image_info']
                    file_id = image_info['id']
                    image_url = create_public_drive_link(file_id)

                    if image_url:
                        line_bot_api.push_message(
                            user_id,
                            ImageSendMessage(
                                original_content_url=image_url,
                                preview_image_url=image_url
                            )
                        )
                        time.sleep(1)  # avoid LINE rate limit

        else:
            flex_message = create_result_flex_message(None, message)
            line_bot_api.push_message(user_id, flex_message)

    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")
        )

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


def test_google_drive_connection():
    """ทดสอบการเชื่อมต่อ Google Drive"""
    try:
        service = authenticate_google_drive()
        results = service.files().list(pageSize=1).execute()
        logger.info("✅ เชื่อมต่อ Google Drive สำเร็จ")
        return True
    except Exception as e:
        logger.error(f"❌ ไม่สามารถเชื่อมต่อ Google Drive: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("🌟 เริ่มต้น LINE Bot Server")
    
    # แสดงข้อมูล Service Account
    service_email = show_service_account_info()
    
    # ทดสอบการเชื่อมต่อ Google Drive ก่อนเริ่ม server
    try:
        if test_google_drive_connection():
            logger.info("🚀 เริ่ม Flask server...")
            app.run(host='0.0.0.0', port=5000, debug=False)
        else:
            logger.error("❌ ไม่สามารถเริ่ม server ได้ เนื่องจากปัญหา Google Drive")
            if service_email:
                print(f"\n💡 แชร์โฟลเดอร์ Google Drive ให้กับ: {service_email}")
            print("\n🔧 วิธีแก้ไข:")
            print("1. ตรวจสอบไฟล์ service-account-key.json")
            print("2. ตรวจสอบ GOOGLE_DRIVE_FOLDER_ID")
            print("3. แชร์โฟลเดอร์ให้กับ Service Account email")
    except KeyboardInterrupt:
        logger.info("👋 ปิด server")
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")