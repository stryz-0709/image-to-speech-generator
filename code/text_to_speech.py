from gtts import gTTS
import os


def generate_audio(caption_text, output_filepath, language='en', slow=False):
    if not caption_text or not caption_text.strip():
        return False
    
    try:
        tts = gTTS(text=caption_text, lang=language, slow=slow)
        output_dir = os.path.dirname(output_filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        tts.save(output_filepath)
        return os.path.exists(output_filepath)
    except Exception:
        return False
