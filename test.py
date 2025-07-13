from find_most_similar_face import find_most_similar_face

user_image = 'piano2.jpg'              # รูปที่ผู้ใช้ส่งมา
folder_path = 'images_from_drive'          # โฟลเดอร์ที่เก็บรูปจาก Google Drive

# เรียกใช้งานฟังก์ชัน
best_match = find_most_similar_face(user_image, folder_path)

if best_match:
    print(f"\n✅ ใบหน้าที่ใกล้ที่สุดคือ: {best_match}")
else:
    print("\n❌ ไม่พบใบหน้าที่ใกล้เคียงพอ")
