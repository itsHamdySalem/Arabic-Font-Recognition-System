import requests

def upload_image(image_path, actual_prediction=None):
    url = 'https://arabic-fonts-detector.onrender.com/predict'
    
    files = {'image': open(image_path, 'rb')}
    
    response = requests.post(url, files=files)

    if response.status_code == 200:
        data = response.json()
        predicted = data['prediction']
        print("Predicted:", predicted)
    else:
        print("Error:", response.text)

if __name__ == '__main__':
    image_path = 'data/518.jpeg'
    actual_prediction = 0    
    upload_image(image_path)
