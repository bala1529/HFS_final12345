## HFS (Hunt For Solution)

Flask web app for:
- URL checker
- SMS checker (text + OCR)
- AI voice analyser (antispoofing)
- Deepfake image / video detector

### Run locally (Windows)

```powershell
cd C:\HFS
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

### Environment variables

- **VT_API_KEY**: VirusTotal API key (optional, improves URL checks)
- **MAIL_USERNAME / MAIL_PASSWORD**: Gmail + App Password for feedback sending

