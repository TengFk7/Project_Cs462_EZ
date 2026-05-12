from flask import Flask, render_template, request, jsonify, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import base64

# หาตำแหน่งของไฟล์ app.py นี้ก่อน
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# กำหนดว่าโฟลเดอร์ uploads ต้องอยู่ที่เดียวกับ app.py
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# คำสั่งวิเศษ: ถ้ายังไม่มีโฟลเดอร์นี้ ให้สร้างขึ้นมาทันที!
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"สร้างโฟลเดอร์สำเร็จที่: {UPLOAD_FOLDER}")

app = Flask(__name__)
# 1. โหลดโมเดล (ตรวจสอบชื่อไฟล์ให้ตรงกับที่คุณ Save มาจาก Colab)
MODEL_PATH = os.path.join(BASE_DIR, 'thai_digit_model.h5')
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None
    print("Warning: ยังไม่มีไฟล์โมเดล! กรุณาเทรนและนำไฟล์มาวางในโฟลเดอร์")

# รายชื่อคลาส (เรียงตามที่เทรนใน Colab)
class_names = ['๓๑', '๓๒', '๓๓', '๓๔', '๓๕']


@app.route('/')
def user_page():
    # ลบ 'project_folder\templates\' ออกให้หมด เหลือแค่ชื่อไฟล์
    return render_template('index.html')

@app.route('/admin')
def admin_page():
    # ลบ Path ยาวๆ ออกเช่นกัน
    return render_template('admin.html')



@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # รับข้อมูลภาพ Base64 จาก Canvas
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # แปลงข้อมูลเป็นรูปภาพ
        image = Image.open(io.BytesIO(image_bytes)).convert('L') # แปลงเป็นขาวดำ (Grayscale)
        
        # Preprocessing: ปรับขนาดให้เป็น 28x28 เหมือนตอนเทรน
        image = image.resize((28, 28))
        image_array = np.array(image) / 255.0  # Normalization
        image_array = image_array.reshape(1, 28, 28, 1) # Reshape ให้เข้ากับ CNN
        
        # ทำนายผล
        predictions = model.predict(image_array)
        score = np.max(predictions) # ค่าความมั่นใจ
        class_idx = np.argmax(predictions) # คลาสที่ทายได้
        
        return jsonify({
            'prediction': class_names[class_idx],
            'confidence': float(score)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if file:
        filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)
        
        global model
        try:
            model = tf.keras.models.load_model(save_path)
            
            # 2. แก้จาก return ข้อความธรรมดา เป็นให้ดีดกลับไปหน้าหลัก
            # 'user_page' คือชื่อฟังก์ชันของหน้า index ( @app.route('/') )
            return redirect(url_for('user_page'))
            
        except Exception as e:
            return f"Error loading new model: {str(e)}", 500

if __name__ == '__main__':
    # รันบน localhost:5000
    app.run(debug=True)