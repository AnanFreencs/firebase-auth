from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model machine learning
model = joblib.load('diabetes_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    glucose = data['glucose']
    heart_rate = data['heart_rate']
    spo2 = data['spo2']
    
    # Lakukan prediksi
    prediction = model.predict([[glucose, heart_rate, spo2]])
    
    if prediction[0] == 1:
        result = {'diagnosis': 'Positif Diabetes', 'advice': 'Segera konsultasikan dengan dokter.'}
    else:
        result = {'diagnosis': 'Negatif Diabetes', 'advice': 'Tetap jaga pola makan dan lakukan pemeriksaan rutin.'}
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
