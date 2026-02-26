from pathlib import Path
from urllib.parse import urlparse
import logging

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge, HTTPException

from config import Config
from modules import (
    url_checker,
    sms_checker,
    voice_analyser,
    deepfake_image,
    deepfake_video,
    ocr_utils,
)


def configure_logging(app: Flask) -> None:
    if app.logger.handlers:
        # Assume logging is already configured by the environment (e.g. gunicorn)
        return

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    )
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(error: RequestEntityTooLarge):
        app.logger.warning("Uploaded file too large: %s", error)
        flash("Uploaded file is too large (maximum 16 MB).", "error")
        return redirect(url_for("index"))

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        # Preserve standard HTTP errors (404, 405, etc.)
        if isinstance(error, HTTPException):
            return error

        app.logger.exception("Unhandled application error: %s", error)
        flash("An unexpected error occurred. Please try again later.", "error")
        return redirect(url_for("index"))


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    configure_logging(app)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    register_routes(app)
    register_error_handlers(app)
    return app


def is_url(text: str) -> bool:
    if not text:
        return False
    parsed = urlparse(text.strip())
    return all([parsed.scheme in ("http", "https"), parsed.netloc])


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def register_routes(app: Flask) -> None:
    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/analyze", methods=["POST"])
    def analyze():
        try:
            text_input = request.form.get("user_input", "").strip()
            uploaded_file = request.files.get("file")
            image_mode = request.form.get("image_mode")  # "deepfake" or "sms"

            analysis_result = None
            module_name = None
            media_file: str | None = None
            media_type: str | None = None

            # ---- File has priority ----
            if uploaded_file and uploaded_file.filename:
                filename = secure_filename(uploaded_file.filename)
                extension = (
                    filename.rsplit(".", 1)[1].lower() if "." in filename else ""
                )

                if not allowed_file(filename, app.config["ALLOWED_EXTENSIONS"]):
                    flash("Unsupported file type.", "error")
                    return redirect(url_for("index"))

                upload_path = Path(app.config["UPLOAD_FOLDER"]) / filename
                uploaded_file.save(upload_path)

                if extension in app.config["AUDIO_EXTENSIONS"]:
                    module_name = "AI VOICE ANALYSER"
                    analysis_result = voice_analyser.analyze_voice(upload_path)

                elif extension in app.config["VIDEO_EXTENSIONS"]:
                    module_name = "DEEPFAKE VIDEO DETECTOR"
                    analysis_result = deepfake_video.analyze_video(upload_path)
                    media_file = filename
                    media_type = "video"

                elif extension in app.config["IMAGE_EXTENSIONS"]:
                    if image_mode == "deepfake":
                        module_name = "DEEPFAKE IMAGE DETECTOR"
                        analysis_result = deepfake_image.analyze_image(upload_path)
                        media_file = filename
                        media_type = "image"
                    else:
                        module_name = "SMS CHECKER (OCR FROM IMAGE)"
                        extracted_text = ocr_utils.extract_text_from_image(upload_path)
                        if ocr_utils.is_ocr_failed(extracted_text):
                            analysis_result = {
                                "risk": 0.0,
                                "confidence": 0.0,
                                "details": (
                                    "Could not extract text from image (OCR failed).\n\n"
                                    "• Install Tesseract OCR and add it to your PATH.\n"
                                    "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
                                    "  After install, add e.g. C:\\Program Files\\Tesseract-OCR to PATH.\n\n"
                                    "• Or use a clearer / higher-resolution image and try again."
                                ),
                                "status": "Unavailable",
                            }
                        elif ocr_utils.is_no_text(extracted_text):
                            analysis_result = {
                                "risk": 0.0,
                                "confidence": 0.0,
                                "details": (
                                    "No readable text was detected in the uploaded image.\n\n"
                                    "Try:\n"
                                    "- A clearer / higher-resolution image\n"
                                    "- Better lighting and straight (non-tilted) capture\n"
                                    "- Cropping to only the message area"
                                ),
                                "status": "No Text",
                            }
                        else:
                            sms_result = sms_checker.analyze_text(extracted_text)
                            sms_result["prefill_text"] = extracted_text
                            analysis_result = sms_result
                else:
                    flash("Unknown file type.", "error")
                    return redirect(url_for("index"))

            # ---- Text only ----
            elif text_input:
                if is_url(text_input):
                    module_name = "URL CHECKER"
                    analysis_result = url_checker.analyze_url(text_input)
                else:
                    module_name = "SMS CHECKER"
                    analysis_result = sms_checker.analyze_text(text_input)
                    analysis_result["prefill_text"] = text_input

            else:
                flash("Please provide a URL / text or upload a file.", "error")
                return redirect(url_for("index"))

            if not analysis_result:
                flash("Analysis failed. Please try again.", "error")
                return redirect(url_for("index"))

            risk = float(analysis_result.get("risk", 0.0))
            confidence = float(analysis_result.get("confidence", 0.0))
            details = str(analysis_result.get("details", "No details available."))
            status_override = analysis_result.get("status")
            prefill_text = analysis_result.get("prefill_text")

            # OCR failed / Unavailable: show as-is (e.g. "Unavailable")
            if status_override == "Unavailable":
                status = "Unavailable"
                display_risk = risk
            elif status_override == "No Text":
                status = "No Text"
                display_risk = risk
            # SMS checker: use 20% threshold for Fake/Dangerous and boost percentage
            elif module_name and "SMS" in module_name and risk > 20:
                status = "Dangerous" if risk >= 60 else "Fake"
                display_risk = min(100.0, risk + 35.0)
            # AI voice analyser: treat low spoof probability as fully safe and clamp to 0.
            elif module_name == "AI VOICE ANALYSER" and risk < 35:
                status = "Safe"
                display_risk = 0.0
            else:
                # Other modules: 35% threshold
                if risk >= 60:
                    status = "Dangerous"
                elif risk >= 35:
                    status = "Fake"
                elif risk >= 15:
                    status = "Low Risk"
                else:
                    status = "Safe"
                display_risk = min(100.0, risk + 35.0) if risk >= 35 else risk

            # For deepfake modules, adjust wording:
            # - "Low Risk" -> "Real"
            # - "Dangerous" -> "Fake"
            display_status = status
            if module_name and "DEEPFAKE" in module_name:
                if status == "Low Risk":
                    display_status = "Real"
                elif status == "Dangerous":
                    display_status = "Fake"

            return render_template(
                "result.html",
                module_name=module_name,
                risk=round(display_risk, 2),
                confidence=round(confidence, 2),
                details=details,
                status=display_status,
                prefill_text=prefill_text,
                heatmap_file=analysis_result.get("heatmap_file"),
                frame_probs=analysis_result.get("frame_probs"),
                media_file=media_file,
                media_type=media_type,
            )
        except Exception as exc:
            app.logger.exception("Error during analysis: %s", exc)
            flash(
                "An unexpected error occurred while processing your input. "
                "Please try again.",
                "error",
            )
            return redirect(url_for("index"))

    @app.route("/disclaimer")
    def disclaimer():
        return render_template("disclaimer.html")

    @app.route("/notes")
    def notes():
        return render_template("notes.html")

    @app.route("/feedback", methods=["GET", "POST"])
    def feedback():
        from smtplib import SMTP_SSL
        from email.message import EmailMessage

        if request.method == "POST":
            email = request.form.get("email", "").strip()
            message = request.form.get("message", "").strip()

            if not email:
                flash("Email is required.", "error")
                return redirect(url_for("feedback"))

            if not message:
                flash("Feedback message cannot be empty.", "error")
                return redirect(url_for("feedback"))

            try:
                msg = EmailMessage()
                msg["Subject"] = "New HFS Feedback"
                msg["From"] = app.config["MAIL_USERNAME"]
                # Always deliver feedback to the HFS inbox.
                msg["To"] = "huntforsolution@gmail.com"
                msg.set_content(f"From: {email}\n\n{message}")

                with SMTP_SSL("smtp.gmail.com", 465) as smtp:
                    smtp.login(app.config["MAIL_USERNAME"], app.config["MAIL_PASSWORD"])
                    smtp.send_message(msg)

                flash("Thank you for your feedback!", "success")
                return redirect(url_for("feedback"))
            except Exception:
                flash("Failed to send feedback. Please try again later.", "error")

        return render_template("feedback.html")

    @app.route("/contact")
    def contact():
        return render_template("contact.html")


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)