import requests

file_path = r'c:\Users\lenovo\Desktop\Auto ML\AvgRate.csv'

with open(file_path, 'r', encoding='iso-8859-1') as file:
    csv_content = file.read()

resp = requests.post("http://localhost:5000/predict", files={'file': csv_content})
print(resp.text)
