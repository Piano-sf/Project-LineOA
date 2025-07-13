import os
import cv2
import numpy as np
import logging
import json
from datetime import datetime
from tqdm import tqdm
from insightface.app import FaceAnalysis

# === กำหนด threshold ===
COSINE_SIM_THRESHOLD = 0.4

# === ตั้งค่า logging ===
def setup_logging():
    """ตั้งค่า logging system"""
    # สร้างโฟลเดอร์ logs ถ้ายังไม่มี
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # ตั้งค่า logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # สร้าง timestamp สำหรับชื่อไฟล์
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/face_similarity_{timestamp}.log'
    
    # ตั้งค่า logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # แสดงใน console ด้วย
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 เริ่มต้นการทำงาน - Log file: {log_filename}")
    return logger

# === โหลดโมเดล antelopev2 ===
logger = setup_logging()
logger.info("📥 กำลังโหลดโมเดล InsightFace...")

try:
    face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("✅ โหลดโมเดล InsightFace สำเร็จ")
except Exception as e:
    logger.error(f"❌ โหลดโมเดลไม่สำเร็จ: {str(e)}")
    raise

def cosine_similarity(a, b):
    """คำนวณ cosine similarity ระหว่างเวกเตอร์ 2 ตัว"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        logger.warning("⚠️ พบ zero vector ในการคำนวณ cosine similarity")
        return 0.0
    
    similarity = np.dot(a, b) / (norm_a * norm_b)
    return similarity

def preprocess_image(image_path):
    """โหลดและปรับแต่งภาพก่อนประมวลผล"""
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"❌ ไม่สามารถโหลดภาพได้: {image_path}")
        return None
    
    height, width = img.shape[:2]
    original_size = (width, height)
    
    # ปรับขนาดภาพถ้าใหญ่เกินไป
    if width > 1024 or height > 1024:
        scale = min(1024/width, 1024/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
        logger.info(f"📐 ปรับขนาดภาพ {image_path}: {original_size} -> {(new_width, new_height)}")
    
    return img

def get_main_face_embedding(image_path):
    """โหลดภาพ และคืน embedding ของใบหน้าหลัก (ใหญ่สุด)"""
    logger.debug(f"🔍 กำลังประมวลผล: {image_path}")
    
    img = preprocess_image(image_path)
    if img is None:
        return None

    try:
        faces = face_app.get(img)
        logger.debug(f"👤 พบใบหน้า {len(faces)} ใบในภาพ: {image_path}")
    except Exception as e:
        logger.error(f"⚠️ เกิดข้อผิดพลาดในการประมวลผล {image_path}: {str(e)}")
        return None
    
    if not faces:
        logger.warning(f"⚠️ ไม่พบใบหน้าในภาพ: {image_path}")
        return None

    # เลือกใบหน้าที่ใหญ่ที่สุด และมี confidence สูง
    valid_faces = [f for f in faces if hasattr(f, 'det_score') and f.det_score > 0.5]
    if not valid_faces:
        valid_faces = faces
        logger.debug(f"🔄 ใช้ใบหน้าทั้งหมด (ไม่มี det_score): {image_path}")
    else:
        logger.debug(f"✅ กรองใบหน้าที่มี confidence > 0.5: {len(valid_faces)}/{len(faces)}")
    
    main_face = max(valid_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    
    # Log ข้อมูลใบหน้า
    face_area = (main_face.bbox[2] - main_face.bbox[0]) * (main_face.bbox[3] - main_face.bbox[1])
    confidence = getattr(main_face, 'det_score', 'N/A')
    logger.debug(f"📊 ใบหน้าหลัก - Area: {face_area:.0f}px², Confidence: {confidence}")
    
    if main_face.embedding is None or len(main_face.embedding) == 0:
        logger.error(f"❌ ไม่สามารถสร้าง embedding ได้: {image_path}")
        return None
    
    logger.debug(f"✅ สร้าง embedding สำเร็จ: {image_path} (dimension: {len(main_face.embedding)})")
    return main_face.embedding

def is_valid_image_file(filename):
    """ตรวจสอบว่าเป็นไฟล์ภาพที่ถูกต้องหรือไม่"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def save_results_to_json(results, user_image, folder_path, best_match, best_score):
    """บันทึกผลลัพธ์เป็น JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f'logs/results_{timestamp}.json'
    
    result_data = {
        'timestamp': datetime.now().isoformat(),
        'user_image': user_image,
        'search_folder': folder_path,
        'threshold': COSINE_SIM_THRESHOLD,
        'best_match': {
            'image': best_match,
            'score': float(best_score) if best_score is not None else None
        },
        'all_results': [
            {
                'image': fname,
                'score': float(score)
            } for fname, score in results
        ],
        'total_images_processed': len(results)
    }
    
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 บันทึกผลลัพธ์ใน: {json_filename}")
    except Exception as e:
        logger.error(f"❌ ไม่สามารถบันทึกผลลัพธ์ JSON: {str(e)}")

def find_most_similar_face(user_image_path, folder_path):
    """เปรียบเทียบใบหน้าผู้ใช้กับภาพทั้งหมดในโฟลเดอร์"""
    logger.info(f"🎯 เริ่มการค้นหา - User: {user_image_path}, Folder: {folder_path}")
    
    # ตรวจสอบว่าไฟล์และโฟลเดอร์มีอยู่จริง
    if not os.path.exists(user_image_path):
        logger.error(f"❌ ไม่พบไฟล์ภาพผู้ใช้: {user_image_path}")
        return None
    
    if not os.path.exists(folder_path):
        logger.error(f"❌ ไม่พบโฟลเดอร์: {folder_path}")
        return None
    
    # ประมวลผลภาพผู้ใช้
    logger.info("👤 กำลังประมวลผลภาพผู้ใช้...")
    user_embedding = get_main_face_embedding(user_image_path)
    if user_embedding is None:
        logger.error("❌ ไม่พบใบหน้าในภาพผู้ใช้")
        return None
    
    logger.info("✅ ประมวลผลภาพผู้ใช้สำเร็จ")

    # กรองไฟล์ภาพ
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if is_valid_image_file(f)]
    
    logger.info(f"📁 ไฟล์ทั้งหมด: {len(all_files)}, ไฟล์ภาพ: {len(image_files)}")
    
    if not image_files:
        logger.error("❌ ไม่พบไฟล์ภาพในโฟลเดอร์")
        return None

    # เริ่มการเปรียบเทียบ
    best_score = -1.0
    best_image = None
    results = []
    processed_count = 0
    failed_count = 0

    logger.info(f"🔍 เริ่มเปรียบเทียบ {len(image_files)} ไฟล์...")
    
    for fname in tqdm(image_files, desc="🔍 กำลังเปรียบเทียบ"):
        image_path = os.path.join(folder_path, fname)
        target_embedding = get_main_face_embedding(image_path)
        
        if target_embedding is None:
            failed_count += 1
            continue

        score = cosine_similarity(user_embedding, target_embedding)
        results.append((fname, score))
        processed_count += 1
        
        logger.debug(f"📊 {fname}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_image = image_path
            logger.debug(f"🏆 ใหม่ที่ดีที่สุด: {fname} ({score:.4f})")

    # สรุปผลลัพธ์
    logger.info(f"📈 สรุปการประมวลผล:")
    logger.info(f"  - ประมวลผลสำเร็จ: {processed_count}/{len(image_files)}")
    logger.info(f"  - ประมวลผลไม่สำเร็จ: {failed_count}/{len(image_files)}")
    logger.info(f"  - คะแนนสูงสุด: {best_score:.4f}")
    logger.info(f"  - Threshold: {COSINE_SIM_THRESHOLD}")

    # เรียงลำดับและแสดงผล top 5
    results.sort(key=lambda x: x[1], reverse=True)
    logger.info("\n📊 Top 5 ผลลัพธ์:")
    for i, (fname, score) in enumerate(results[:5]):
        logger.info(f"  {i+1}. {fname}: {score:.4f}")

    # บันทึกผลลัพธ์เป็น JSON
    save_results_to_json(results, user_image_path, folder_path, best_image, best_score)

    # ตัดสินใจผลลัพธ์
    if best_score >= COSINE_SIM_THRESHOLD:
        logger.info(f"✅ เจอภาพที่ใกล้ที่สุด: {best_image} (similarity: {best_score:.4f})")
        return best_image
    else:
        logger.warning(f"❌ ไม่มีใบหน้าที่ใกล้เคียงพอ (similarity สูงสุด = {best_score:.4f})")
        logger.info(f"💡 ลองลด threshold จาก {COSINE_SIM_THRESHOLD} เป็น {best_score:.2f}")
        return None

# === ตัวอย่างการใช้งาน ===
if __name__ == "__main__":
    user_image = "IMG_0472.jpg"
    image_folder = "images_from_drive"
    
    logger.info("=" * 60)
    logger.info("🚀 เริ่มต้นโปรแกรม Face Similarity")
    logger.info(f"📸 ภาพผู้ใช้: {user_image}")
    logger.info(f"📁 โฟลเดอร์ค้นหา: {image_folder}")
    logger.info(f"🎯 Threshold: {COSINE_SIM_THRESHOLD}")
    logger.info("=" * 60)

    match = find_most_similar_face(user_image, image_folder)
    
    if match:
        logger.info(f"🎉 ผลลัพธ์สุดท้าย: {match}")
    else:
        logger.info("💭 ข้อเสนอแนะ:")
        logger.info("  1. ตรวจสอบว่าภาพผู้ใช้และภาพในโฟลเดอร์มีใบหน้าที่ชัดเจน")
        logger.info("  2. ลองปรับ threshold ให้ต่ำลง")
        logger.info("  3. ตรวจสอบคุณภาพของภาพ (ความสว่าง, ความคมชัด)")
    
    logger.info("=" * 60)
    logger.info("🏁 สิ้นสุดการทำงาน")
    logger.info("=" * 60)