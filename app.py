# Importing essential libraries and modules
from flask import Flask, render_template, Markup
import pandas as pd
import Disease_Dictionary
from flask import request
from flask import Flask, render_template, jsonify, request, Markup
from disease_model import predict_image
import csv
import numpy as np

# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    title = 'Plant Disease Detection'
    return render_template('landing.html', title=title)

#########Post Functions##########

@app.route('/result', methods=['POST'])
def disease_predict():
    if request.method == 'POST':
        try:
            img = request.files['file']
            # img = file.read()
            # Image is in bytes till here
            img_path = img.filename
            # print(img_path, type(img_path))
            img.save(f'static/user uploaded/{img_path}')
            first, second, third, max_values = predict_image(f'static/user uploaded/{img_path}')
            first_confidence = np.round(max_values[2]*100, 2)
            second_confidence = np.round(max_values[1]*100, 2)
            third_confidence = np.round(max_values[0]*100, 2)
            print(first_confidence, second_confidence, third_confidence)
            with open('Disease_Dictionary/Disease_dictionary.csv') as file_obj:
                reader_obj = csv.DictReader(file_obj)
                for row in reader_obj:
                    if first != 'Objects' and second != 'Objects' and third != 'Objects':
                        if row['Prediction'] == first:
                            crop_1 = Markup(row['Crop'])
                            disease_1 = Markup(row['Disease'])
                            print(crop_1)
                            print(disease_1)
                        if row['Prediction'] == second:
                            crop_2 = Markup(row['Crop'])
                            disease_2 = Markup(row['Disease'])
                            print(crop_2)
                            print(disease_2)
                        if row['Prediction'] == third:
                            crop_3 = Markup(row['Crop'])
                            disease_3 = Markup(row['Disease'])
                            print(crop_3)
                            print(disease_3)
                    # elif first == 'Objects' or second == 'Objects' or third == 'Objects':
                        # crop_4 = Markup(row['Crop'])
                        # disease_4 = Markup(row['Disease'])
                    else:
                        crop_4 = 'Object'
                        disease_4 = 'Not a crop'
                        print(crop_4)
                        print(disease_4)
                        return render_template('object_result.html', status=200, crop_4=crop_4, disease_4=disease_4,
                                               first_confidence=first_confidence, img=f'static/user uploaded/{img_path}')

            return render_template('disease_result.html', status=200, crop_1=crop_1, disease_1=disease_1,
                                   crop_2=crop_2, disease_2=disease_2, crop_3=crop_3, disease_3=disease_3,
                                   first_confidence=first_confidence, second_confidence=second_confidence,
                                   third_confidence=third_confidence, img=f'static/user uploaded/{img_path}')
        except:
            pass
    return render_template('disease_result.html', status=500, res="Internal Server Error")

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)


