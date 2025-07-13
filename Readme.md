# LINE Bot Face Recognition System

A sophisticated LINE Bot that uses AI-powered face recognition to find matching faces in your Google Drive photo collection. Built with InsightFace, Google Drive API, and LINE Bot SDK.

## Features

- 🤖 **AI Face Recognition**: Uses InsightFace's buffalo_l model for accurate face detection and matching
- 📁 **Google Drive Integration**: Automatically searches through your Google Drive photo collection
- 💬 **LINE Bot Interface**: Easy-to-use chat interface for uploading and searching faces
- 🎯 **High Accuracy**: Configurable similarity threshold for precise matching
- 📊 **Detailed Logging**: Comprehensive logging system for monitoring and debugging
- 🔒 **Secure**: OAuth2 authentication with Google Drive API

## Prerequisites

- Python 3.9 or higher
- LINE Developer Account
- Google Cloud Platform Account
- Google Drive API enabled

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd line-bot-face-recognition
```

### 2. Run Setup Script
```bash
python setup.py
```

### 3. Configure Environment
Edit `.env` file with your actual configuration:
```env
LINE_CHANNEL_ACCESS_TOKEN=your_channel_access_token_here
LINE_CHANNEL_SECRET=your_channel_secret_here
GOOGLE_DRIVE_FOLDER_ID=your_google_drive_folder_id_here
```

### 4. Add Google Credentials
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google Drive API
4. Create OAuth 2.0 credentials
5. Download `credentials.json` and place it in the project root

### 5. Run the Application
```bash
python app.py
```

## Manual Installation

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Create Required Directories
```bash
mkdir -p logs temp data
```

### Setup Google Authentication
```bash
python -c "from google_auth_oauthlib.flow import InstalledAppFlow; flow = InstalledAppFlow.from_client_secrets_file('credentials.json', ['https://www.googleapis.com/auth/drive.readonly']); flow.run_local_server(port=0)"
```

## Docker Deployment

### Build and Run with Docker Compose
```bash
docker-compose up -d
```

### Build Docker Image Manually
```bash
docker build -t line-bot-face-recognition .
docker run -p 5000:5000 -v $(pwd)/logs:/app/logs line-bot-face-recognition
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LINE_CHANNEL_ACCESS_TOKEN` | LINE Bot channel access token | Required |
| `LINE_CHANNEL_SECRET` | LINE Bot channel secret | Required |
| `GOOGLE_DRIVE_FOLDER_ID` | Google Drive folder ID to search | Required |
| `COSINE_SIM_THRESHOLD` | Face similarity threshold (0.0-1.0) | 0.4 |
| `FLASK_PORT` | Server port | 5000 |
| `FLASK_HOST` | Server host | 0.0.0.0 |
| `LOG_LEVEL` | Logging level | INFO |

### Face Recognition Settings

- **Model**: InsightFace buffalo_l
- **Detection Size**: 640x640 pixels
- **Similarity Threshold**: 0.4 (adjustable)
- **Max Image Size**: 1024x1024 pixels

## Usage

### LINE Bot Commands

1. **Send a Photo**: Upload a photo with a face to search for similar faces
2. **Help**: Type `help`, `ช่วยเหลือ`, or `วิธีใช้` for usage instructions

### API Endpoints

- `POST /callback` - LINE Bot webhook endpoint
- `GET /health` - Health check endpoint (for Docker)

## How It Works

1. **Face Detection**: Uses InsightFace to extract face embeddings from uploaded images
2. **Google Drive Search**: Searches through specified Google Drive folder for images
3. **Similarity Matching**: Compares face embeddings using cosine similarity
4. **Result Delivery**: Returns matching images via LINE Bot interface

## Logging

The application creates detailed logs in the `logs/` directory:
- Timestamp-based log files
- Face detection results
- Similarity scores
- Error tracking

## Security Considerations

- Google Drive credentials are stored locally
- OAuth2 tokens are refreshed automatically
- LINE Bot signatures are verified
- No face data is permanently stored

## Troubleshooting

### Common Issues

1. **InsightFace Model Loading Error**
   - Ensure sufficient system memory
   - Check ONNX runtime installation

2. **Google Drive Authentication Failed**
   - Verify `credentials.json` is correct
   - Check Google Drive API is enabled
   - Ensure proper OAuth2 scope permissions

3. **LINE Bot Webhook Not Working**
   - Verify webhook URL is accessible
   - Check LINE Bot channel configuration
   - Ensure proper SSL certificate

### Debug Mode

Enable debug logging:
```env
LOG_LEVEL=DEBUG
FLASK_DEBUG=True
```

## Performance Optimization

- **Image Resizing**: Automatically resizes large images
- **Efficient Processing**: Processes only face regions
- **Memory Management**: Clears temporary data after processing
- **Batch Processing**: Handles multiple images efficiently

## System Requirements

### Minimum Requirements
- RAM: 4GB
- CPU: 2 cores
- Storage: 10GB free space

### Recommended Requirements
- RAM: 8GB or more
- CPU: 4 cores or more
- Storage: 50GB free space
- GPU: Optional, for faster processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/` directory
3. Open an issue on GitHub

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for face recognition
- [LINE Bot SDK](https://github.com/line/line-bot-sdk-python) for LINE integration
- [Google Drive API](https://developers.google.com/drive) for cloud storage