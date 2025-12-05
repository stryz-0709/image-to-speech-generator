import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
from image_to_text import generate_caption, initialize_models
from text_to_speech import generate_audio
from helper.file_utils import validate_image_file, check_extension

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')
app.config['AUDIO_FOLDER'] = os.path.join(app.root_path, 'static/audio')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

try:
    initialize_models()
except Exception:
    pass


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method != 'POST':
        return render_template('index.html', 
            pretrained_caption=None, 
            fine_tuned_caption=None, 
            pretrained_audio_url=None, 
            fine_tuned_audio_url=None, 
            image_url=None, 
            error=None)
    
    if 'file' not in request.files or not request.files['file'].filename:
        flash('No file selected', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    if not check_extension(file.filename):
        flash('Invalid file type', 'error')
        return redirect(request.url)
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if not validate_image_file(filepath)[0]:
            os.remove(filepath)
            flash('Invalid image file', 'error')
            return redirect(request.url)
        
        img = Image.open(filepath)
        pretrained_caption, fine_tuned_caption = generate_caption(img)
        
        def _audio_url(caption, suffix):
            if not caption:
                return None
            audio_filename = f"{os.path.splitext(filename)[0]}_{suffix}.mp3"
            audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
            return url_for('static', filename=f'audio/{audio_filename}') if generate_audio(caption, audio_path) else None
        
        return render_template('index.html',
            pretrained_caption=pretrained_caption,
            fine_tuned_caption=fine_tuned_caption,
            pretrained_audio_url=_audio_url(pretrained_caption, 'pretrained'),
            fine_tuned_audio_url=_audio_url(fine_tuned_caption, 'finetuned'),
            image_url=url_for('static', filename=f'uploads/{filename}'),
            error=None)
            
    except Exception:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        flash('Error processing file', 'error')
        return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
