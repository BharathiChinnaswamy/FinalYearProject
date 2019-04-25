from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import os
import reinforcement
app = Flask(__name__,  template_folder='templates')

UPLOAD_FOLDER = 'preprocess'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("model.html")


@app.route("/upload/<modelname>", methods=["GET", "POST"])
def upload(modelname):
    if request.method == 'POST':
        f = request.files['fileToUpload']

        f.save(os.path.join("prediction", secure_filename(f.filename)))
        return redirect(url_for('predict', modelname=modelname, filename=secure_filename(f.filename)))

    return render_template("upload_new.html")


@app.route("/predict/<modelname>/<filename>", methods=["GET", "POST"])
def predict(modelname, filename):
    y = reinforcement.predict(filename)
    print(y)
    return render_template("show_steps.html", y=y, filename=".".join(filename.split(".")[:-1]), extension=filename.split(".")[-1],
                           length=len(y))


@app.route('/image/<filename>')
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
