from flask import Flask, request, jsonify
from model import transform_data, get_prediction,rmse
import csv
import os 
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['FILE_UPLOADS'] = ''

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']

        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
       
        
        filepath = os.path.join(app.config['FILE_UPLOADS'], file.filename)
        file.save(filepath)
        with open(filepath,"r",encoding='utf-8') as file:
          csv_file = csv.reader(file)
          print("Received file:", csv_file)  
          
           

        X,Y = transform_data(filepath, 3) 

    prediction,real = get_prediction(X),Y
    numpy_array_prediction,numpy_array_real = prediction.detach().numpy(),real.detach().numpy()
    python_list_prediction,python_list_real = numpy_array_prediction.tolist(),numpy_array_real.tolist()
    RMSE=rmse(prediction,real)
    numpy_array_rmse=RMSE.detach().numpy()
    list_rmse=numpy_array_rmse.tolist()


    data = {'prediction': python_list_prediction,'real':python_list_real,'rmse':list_rmse}
    return jsonify(data)
        

    

if __name__ == "__main__":
    app.run(debug=True)








