import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the LINE Bot application"""
    
    # LINE Bot Configuration
    LINE_CHANNEL_ACCESS_TOKEN = os.getenv('oA4uOEDW7hKpmkO94X0GiSPZ5Kej0QM8vbkCglUC0HLRG/OF9j35XOGTdx0B+bH2rC+70ajDn5jSrj2SBixirpuifvbIBl0Zz48AUMdIWrOPS90B8/IhIF7hs8L5svaRTA82nCGQhfCx2j09JNXuzwdB04t89/1O/w1cDnyilFU=')
    LINE_CHANNEL_SECRET = os.getenv('603e1fd9cb40d7fd19de468c824cacd5')
    
    # Google Drive Configuration
    GOOGLE_DRIVE_FOLDER_ID = os.getenv('1BNmuKRL1vQE2czc9snVg48-S7O6fAjSd')
    GOOGLE_CREDENTIALS_FILE = 'credentials.json'
    GOOGLE_TOKEN_FILE = 'token.pickle'
    
    # Face Recognition Configuration
    COSINE_SIM_THRESHOLD = float(os.getenv('COSINE_SIM_THRESHOLD', '0.4'))
    FACE_MODEL_NAME = 'buffalo_l'
    FACE_DETECTION_SIZE = (640, 640)
    MAX_IMAGE_SIZE = 1024
    
    # Server Configuration
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR = os.getenv('LOG_DIR', 'logs')
    
    # Google Drive API Scopes
    GOOGLE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    @classmethod
    def validate(cls):
        """Validate required configuration values"""
        required_configs = [
            'LINE_CHANNEL_ACCESS_TOKEN',
            'LINE_CHANNEL_SECRET',
            'GOOGLE_DRIVE_FOLDER_ID'
        ]
        
        missing_configs = []
        for config in required_configs:
            if not getattr(cls, config):
                missing_configs.append(config)
        
        if missing_configs:
            raise ValueError(f"Missing required configuration: {', '.join(missing_configs)}")
        
        return True