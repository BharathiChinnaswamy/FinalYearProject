from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import os
app = Flask(__name__,  template_folder='templates')

@app.route("/")
def home():
    return render_template("model.html")

@app.route("/upload/<modelname>", methods=["GET", "POST"])
def upload(modelname):
    if request.method == 'POST':
      f = request.files['fileToUpload']

      f.save(os.path.join("prediction",secure_filename(f.filename)))
      return redirect(url_for('predict', modelname=modelname, filename=secure_filename(f.filename)))
    return render_template("upload_new.html")

@app.route("/predict/<modelname>/<filename>", methods=["GET", "POST"])
def predict(modelname,filename):
    return 'file uploaded successfully'

if __name__ == "__main__":
    app.run(debug=True)
