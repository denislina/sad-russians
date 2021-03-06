import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import imageio
import subprocess
import sys
sys.path.append('../')

MAX_FILE_SIZE = 1024 * 1024 + 1

app = Flask(__name__)

UPLOAD_FOLDER = '../examples'
ALLOWED_EXTENSIONS = set([ 'jpg', 'jpeg', 'JPG', 'JPEG'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_filename(name, extension):
	return '{}.{}'.format(name, extension)


def create_gif(filenames, path=UPLOAD_FOLDER):
	images = []
	for filename in filenames:
	    images.append(imageio.imread('{}/{}'.format(path, filename)))
	imageio.mimsave('{}/result.gif'.format(path), images)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            name, extension = filename.split('.')
            file.save('{}/{}'.format(UPLOAD_FOLDER, secure_filename(file.filename)))
           
            subprocess.call('../src/build/ChangeEmotion --happy {}/{} '.format(UPLOAD_FOLDER, filename),  shell=True)
            new_filename = '{}_happy.jpeg'.format(name)
            create_gif([filename, new_filename])
            new_filename = 'result.gif'
            return redirect(url_for('uploaded_file',
                                    filename=new_filename))
    return render_template('title.html')


@app.route("/", methods=["POST", "GET"])
def index():
    args = {"method": "GET"}
    if request.method == "POST":
        file = request.files["file"]
        if bool(file.filename):
            file_bytes = file.read(MAX_FILE_SIZE)
            args["file_size_error"] = len(file_bytes) == MAX_FILE_SIZE
        args["method"] = "POST"
    return render_template("loader.html", args=args)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=True)
