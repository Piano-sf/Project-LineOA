import os
import io
from tqdm import tqdm
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# === ตั้งค่า ===
SERVICE_ACCOUNT_FILE = 'line-oa-face-match-cb08a498ec3b.json'         # เปลี่ยน path ถ้าเก็บไว้ที่อื่น
FOLDER_ID = '1BNmuKRL1vQE2czc9snVg48-S7O6fAjSd'       # ดูวิธีหาได้ด้านล่าง
DEST_FOLDER = 'images_from_drive'                 # โฟลเดอร์ที่โหลดภาพลง

# === เตรียมโฟลเดอร์ปลายทาง ===
os.makedirs(DEST_FOLDER, exist_ok=True)

# === สร้าง credentials และเชื่อมต่อ API ===
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

drive_service = build('drive', 'v3', credentials=creds)

# === สร้าง query ค้นหาเฉพาะไฟล์ภาพในโฟลเดอร์ ===
query = f"'{FOLDER_ID}' in parents and mimeType contains 'image/' and trashed = false"
results = drive_service.files().list(q=query, pageSize=1000,
                                     fields="files(id, name)").execute()
items = results.get('files', [])

# === ดาวน์โหลดภาพทั้งหมด ===
if not items:
    print("❌ ไม่พบไฟล์ในโฟลเดอร์")
else:
    print(f"📦 เจอ {len(items)} ไฟล์ กำลังดาวน์โหลด...")

    for file in tqdm(items):
        file_id = file['id']
        file_name = file['name']
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        with open(os.path.join(DEST_FOLDER, file_name), 'wb') as f:
            f.write(fh.getbuffer())

    print("✅ ดาวน์โหลดเสร็จแล้ว")
