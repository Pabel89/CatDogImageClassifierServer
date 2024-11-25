"""
Dieses Modul ist ein Bild Uploadserver fuer den Katzen und Hunde Bildklassifizierer
um mit einer clientseitigen App drauf zugreifen zu koennen.
Durch den Upload eines Katzen- oder Hundebildes erhaelt der Client eine Response,
ob das Bild einen Hund oder eine Katze zeigt

Author: Alexander Pabel
"""
import random
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from image_classifier_CNN import classifyImage
from image_classifier_pretrained_VGG16 import classifyImage as classifyImage2

whichClassifierCNN=""
app = Flask(__name__)
CORS(app)  # Erlaube CORS von überall. Ändern im Productionmode

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Kein Teil einer Datei'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Keine Datei selektiert'}), 400

    # Check for file extension (optional)
    allowed_extensions = ['jpg', 'jpeg', 'png']
    if file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Das Dateiformat wird nicht unterstuetzt'}), 400

    # Create uploads directory if it doesn't exist
    try:
        os.makedirs('uploads', exist_ok=True)  # Create uploads directory if needed
    except OSError as e:
        return jsonify({'error': f'Uploadsordner konnte nicht erstellt werden: {e}'}), 500

    # Save the file with a secure filename
    from werkzeug.utils import secure_filename
    random_number = random.randint(1, 1000000)
    filename = secure_filename(file.filename)
    randomintfilename = str(random_number)+'_'+filename
    file.save('uploads/'+ randomintfilename)
    with open('config.json', 'r') as jfile:
        data = json.load(jfile)
        for item in data:
            if item.get('use') == "True":  # Use get() with default value for safe access
                whichClassifierCNN = item['name']
                print("Selected classifier:", whichClassifierCNN)
    if(whichClassifierCNN == "image_classifier_CNN"):
        answer = classifyImage('uploads/' + randomintfilename)
    elif(whichClassifierCNN == "image_classifier_pretrained_VGG16"):
        answer = classifyImage2('uploads/' + randomintfilename)
    # nach der Ueberpruefung wieder loeschen
    if os.path.exists('uploads/' + randomintfilename):
        os.remove('uploads/' + randomintfilename)
        print(f"File '{randomintfilename}' wieder entfernt.")
    else:
        print(f"File '{randomintfilename}' konnte nicht entfernt werden")
    return jsonify({'message': answer})

if __name__ == '__main__':
    app.run(debug=False)