import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# === กำหนด threshold ===
COSINE_SIM_THRESHOLD = 0.7  # ยิ่งมากยิ่งเหมือน (ค่าทั่วไป: 0.5 – 0.8)

# === โหลดโมเดล antelopev2 ===
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

def cosine_similarity(a, b):
    """คำนวณ cosine similarity ระหว่างเวกเตอร์ 2 ตัว"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_main_face_embedding(image_path):
    """โหลดภาพ และคืน embedding ของใบหน้าหลัก (ใหญ่สุด)"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ ไม่สามารถโหลดภาพได้: {image_path}")
        return None

    faces = face_app.get(img)
    if not faces:
        print(f"⚠️ ไม่พบใบหน้าในภาพ: {image_path}")
        return None

    # เลือกใบหน้าที่ใหญ่ที่สุด
    main_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return main_face.embedding

def find_most_similar_face(user_image_path, folder_path):
    """เปรียบเทียบใบหน้าผู้ใช้กับภาพทั้งหมดในโฟลเดอร์"""
    user_embedding = get_main_face_embedding(user_image_path)
    if user_embedding is None:
        print("❌ ไม่พบใบหน้าในภาพผู้ใช้")
        return None

    best_score = -1.0
    best_image = None

    for fname in tqdm(os.listdir(folder_path), desc="🔍 กำลังเปรียบเทียบ"):
        image_path = os.path.join(folder_path, fname)
        target_embedding = get_main_face_embedding(image_path)
        if target_embedding is None:
            continue

        score = cosine_similarity(user_embedding, target_embedding)
        if score > best_score:
            best_score = score
            best_image = image_path

    if best_score >= COSINE_SIM_THRESHOLD:
        print(f"✅ เจอภาพที่ใกล้ที่สุด: {best_image} (similarity: {best_score:.4f})")
        return best_image
    else:
        print(f"❌ ไม่มีใบหน้าที่ใกล้เคียงพอ (similarity สูงสุด = {best_score:.4f})")
        return None

# === ตัวอย่างการใช้งาน ===
if __name__ == "__main__":
    user_image = "piano2.jpg"         # รูปของผู้ใช้
    image_folder = "images_from_drive"      # โฟลเดอร์ที่เก็บภาพทั้งหมด

    match = find_most_similar_face(user_image, image_folder)
    if match:
        print("📸 ภาพที่ตรงกันมากที่สุดคือ:", match)
