from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from classify_model import predict_label

UPLOAD_FOLDER = 'uploads'
SUGGEST_FOLDER = 'user_data'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUGGEST_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contribute', methods=['GET'])
def contribute():
    return render_template('contribute.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'audio' not in request.files:
        return redirect(url_for('index'))
    file = request.files['audio']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    label, confidence = predict_label(filepath)
    os.remove(filepath)
    return render_template('result.html', label=label, confidence=round(confidence * 100, 2))

@app.route('/suggest', methods=['POST'])
def suggest():
    if 'suggest_audio' not in request.files or 'suggest_label' not in request.form:
        return redirect(url_for('index'))

    file = request.files['suggest_audio']
    label = request.form['suggest_label'].strip().lower().replace(' ', '_')
    if file.filename == '' or not allowed_file(file.filename) or not label:
        return redirect(url_for('index'))

    folder = os.path.join(SUGGEST_FOLDER, label)
    os.makedirs(folder, exist_ok=True)
    filename = secure_filename(file.filename)
    file.save(os.path.join(folder, filename))
    return render_template('thankyou.html', label=label)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)