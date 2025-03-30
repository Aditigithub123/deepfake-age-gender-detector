import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import mysql.connector
from tensorflow.keras.layers import InputLayer as KerasInputLayer

# Custom InputLayer to remap 'batch_shape' and fix config issues
class CustomInputLayer(KerasInputLayer):
    def __init__(self, **kwargs):
        # Remap 'batch_shape' to 'batch_input_shape' if present.
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super(CustomInputLayer, self).__init__(**kwargs)
    
    @classmethod
    def from_config(cls, config):
        # If 'batch_shape' exists and is a string, convert it to a list.
        if 'batch_shape' in config and isinstance(config['batch_shape'], str):
            config['batch_shape'] = [config['batch_shape']]
        return super(CustomInputLayer, cls).from_config(config)

app = Flask(__name__)
# Force login every time by clearing the session on logout.
app.secret_key = os.urandom(24)

# MySQL Database Connection configuration
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "Goyal@aditi123",  # Change to your MySQL password
    "database": "age_detection"
}

# Directory for storing uploads
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def load_models():
    # Dictionary of custom objects needed for deserialization.
    custom_objs = {
        'InputLayer': CustomInputLayer,
        'DTypePolicy': tf.keras.mixed_precision.Policy
    }
    # Load models from the "models" folder using relative paths.
    deepfake_model = load_model("models/real_fake_vgg_model.h5", compile=False, custom_objects=custom_objs)
    gender_model = load_model("models/MobileNetV2_gender_detection_model.h5", compile=False, custom_objects=custom_objs)
    age_model = load_model("models/simple_cnn_age.h5", compile=False, custom_objects=custom_objs)
    return deepfake_model, gender_model, age_model

deepfake_model, gender_model, age_model = load_models()

def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image path or file.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, target_size)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def predict(image_path, deepfake_model, gender_model, age_model):
    try:
        deepfake_img_array, _ = preprocess_image(image_path, (224, 224))
        gender_img_array, _ = preprocess_image(image_path, (128, 128))
        age_input_shape = age_model.input_shape[1:3]
        age_img_array, _ = preprocess_image(image_path, age_input_shape)
        
        deepfake_pred = deepfake_model.predict(deepfake_img_array)
        deepfake_label = "Fake" if deepfake_pred[0][0] > 0.4 else "Real"
        
        gender_pred = gender_model.predict(gender_img_array)
        predicted_gender = "Male" if gender_pred[0][0] > 0.4 else "Female"
        
        age_pred = age_model.predict(age_img_array)[0]
        age_mapping = np.array([5, 15, 25, 35, 45, 60])
        normalized_probs = age_pred / np.sum(age_pred)
        age_label = int(np.round(np.sum(normalized_probs * age_mapping)))
        
        return deepfake_label, predicted_gender, age_label
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

def save_base64_image(data, filename):
    header, encoded = data.split(",", 1)
    binary_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(binary_data))
    image.save(filename)

def store_user_data(name, actual_age, gender, image_base64):
    """Store the user data in the MySQL database."""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        sql = "INSERT INTO new_data (name, actual_age, gender, image_data) VALUES (%s, %s, %s, %s)"
        values = (name, actual_age, gender, image_base64)
        cursor.execute(sql, values)
        conn.commit()
        print("User data stored successfully.")
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Always show the login page on the root.
@app.route("/", methods=["GET", "POST"])
def login():
    # Clear any existing session so the login page is always shown.
    session.clear()
    if request.method == "POST":
        name = request.form.get("name")
        actual_age = request.form.get("actual_age")
        gender = request.form.get("gender")
        if not name or not actual_age or not gender:
            flash("Please provide your name, age, and gender.", "error")
            return redirect(url_for("login"))
        # Store in session for use in image processing.
        session["name"] = name
        session["actual_age"] = actual_age
        session["gender"] = gender
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/index", methods=["GET", "POST"])
def index():
    # Ensure the user is logged in via session.
    if "name" not in session or "actual_age" not in session or "gender" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("login"))
    
    name = session.get("name")
    actual_age = session.get("actual_age")
    gender = session.get("gender")
    
    # Debug print the received login data.
    print(f"Login details received: name={name}, actual_age={actual_age}, gender={gender}")
    
    results = None
    img_url = None
    if request.method == "POST":
        # Check if the user submitted an image upload or captured image.
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            with open(filepath, "rb") as f:
                image_bytes = f.read()
            image_base64 = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")
            img_url = url_for("static", filename="uploads/" + filename)
        elif request.form.get("captured_image", None):
            data = request.form.get("captured_image")
            filename = "captured_image.jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            save_base64_image(data, filepath)
            image_base64 = data  # Already in Base64 format.
            img_url = url_for("static", filename="uploads/" + filename)
        else:
            flash("No image provided.", "error")
            return redirect(url_for("index"))
        
        # Debug print to check that login values are received in the POST request as well.
        print("Image processing request received:")
        print("Name:", name)
        print("Actual Age:", actual_age)
        print("Gender:", gender)
        
        deepfake_label, predicted_gender, age_label = predict(filepath, deepfake_model, gender_model, age_model)
        if deepfake_label is None or predicted_gender is None:
            flash("Prediction failed due to errors.", "error")
        else:
            results = {
                "deepfake": deepfake_label,
                "predicted_gender": predicted_gender,
                "age": age_label
            }
            store_user_data(name, actual_age, gender, image_base64)
            flash("User data stored successfully!", "success")
    
    return render_template("index.html", results=results, img_url=img_url, name=name, actual_age=actual_age, gender=gender)

# Optional logout route to clear the session and force login again.
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
