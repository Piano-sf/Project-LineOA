#!/usr/bin/env python3
"""
Setup script for LINE Bot Face Recognition system
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path

class SetupManager:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.required_files = [
            'requirements.txt',
            '.env',
            'credentials.json'
        ]
        
    def create_directories(self):
        """Create necessary directories"""
        directories = ['logs', 'temp', 'data']
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    def install_requirements(self):
        """Install Python requirements"""
        try:
            print("📦 Installing Python requirements...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], check=True)
            print("✅ Requirements installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
        return True
    
    def setup_env_file(self):
        """Setup environment file"""
        env_file = self.project_root / '.env'
        env_example = self.project_root / '.env.example'
        
        if not env_file.exists():
            if env_example.exists():
                shutil.copy(env_example, env_file)
                print("✅ Created .env file from .env.example")
                print("⚠️  Please edit .env file with your actual configuration values")
            else:
                print("❌ .env.example file not found")
                return False
        else:
            print("✅ .env file already exists")
        return True
    
    def check_credentials(self):
        """Check for Google credentials file"""
        credentials_file = self.project_root / 'credentials.json'
        
        if not credentials_file.exists():
            print("❌ credentials.json not found")
            print("📝 Please:")
            print("   1. Go to Google Cloud Console")
            print("   2. Create a new project or select existing one")
            print("   3. Enable Google Drive API")
            print("   4. Create credentials (OAuth 2.0 Client ID)")
            print("   5. Download credentials.json file")
            print("   6. Place it in the project root directory")
            return False
        else:
            print("✅ credentials.json found")
            return True
    
    def test_imports(self):
        """Test if all required modules can be imported"""
        test_modules = [
            'flask',
            'linebot',
            'cv2',
            'numpy',
            'google.oauth2.credentials',
            'googleapiclient.discovery',
            'insightface'
        ]
        
        failed_imports = []
        
        for module in test_modules:
            try:
                __import__(module)
                print(f"✅ {module}")
            except ImportError as e:
                print(f"❌ {module}: {e}")
                failed_imports.append(module)
        
        if failed_imports:
            print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
            return False
        
        print("\n✅ All modules imported successfully")
        return True
    
    def setup_google_auth(self):
        """Setup Google authentication"""
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            import pickle
            
            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            
            creds = None
            token_file = self.project_root / 'token.pickle'
            credentials_file = self.project_root / 'credentials.json'
            
            if not credentials_file.exists():
                print("❌ credentials.json not found. Please add it first.")
                return False
            
            # Load existing token
            if token_file.exists():
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)
            
            # If there are no valid credentials, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(credentials_file), SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(token_file, 'wb') as token:
                    pickle.dump(creds, token)
            
            print("✅ Google authentication setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Google authentication setup failed: {e}")
            return False
    
    def create_systemd_service(self):
        """Create systemd service file for Linux deployment"""
        service_content = f"""[Unit]
Description=LINE Bot Face Recognition Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory={self.project_root}
Environment=PATH={self.project_root}/venv/bin
ExecStart={sys.executable} app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_file = self.project_root / 'line-bot-face-recognition.service'
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print("✅ Created systemd service file")
        print("📝 To install the service:")
        print(f"   sudo cp {service_file} /etc/systemd/system/")
        print("   sudo systemctl daemon-reload")
        print("   sudo systemctl enable line-bot-face-recognition")
        print("   sudo systemctl start line-bot-face-recognition")
    
    def run_setup(self):
        """Run the complete setup process"""
        print("🚀 Starting LINE Bot Face Recognition Setup\n")
        
        # Create directories
        self.create_directories()
        
        # Setup environment file
        if not self.setup_env_file():
            return False
        
        # Install requirements
        if not self.install_requirements():
            return False
        
        # Test imports
        if not self.test_imports():
            return False
        
        # Check credentials
        if not self.check_credentials():
            return False
        
        # Setup Google authentication
        setup_auth = input("\n🔑 Setup Google authentication now? (y/n): ").lower() == 'y'
        if setup_auth:
            self.setup_google_auth()
        
        # Create systemd service
        create_service = input("\n🔧 Create systemd service file? (y/n): ").lower() == 'y'
        if create_service:
            self.create_systemd_service()
        
        print("\n🎉 Setup completed successfully!")
        print("\n📝 Next steps:")
        print("1. Edit .env file with your actual configuration")
        print("2. Add credentials.json file from Google Cloud Console")
        print("3. Run: python app.py")
        
        return True

if __name__ == "__main__":
    setup_manager = SetupManager()
    success = setup_manager.run_setup()
    sys.exit(0 if success else 1)