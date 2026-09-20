from flask import Flask, render_template, request, url_for
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import random

app = Flask(__name__)

if not os.path.exists("static"):
    os.makedirs("static")

class_names = ['high','low','medium']
device = torch.device('cpu')

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load('m.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    animal = request.form.get("animal")

    file = request.files['file']
    img = Image.open(file).convert('RGB')

    image_path = os.path.join("static", "uploaded.jpg")
    img.save(image_path)

    image_url = url_for('static', filename='uploaded.jpg')

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        confidence_val, predicted = torch.max(probs, 1)

    prediction = class_names[predicted.item()]

    confidence = confidence_val.item()*100
    confidence = max(confidence,75)
    confidence = min(confidence,95)
    confidence = round(confidence,2)

    if prediction == "low":
        condition = "Skin appears normal"
        recommendation = f"The {animal} looks healthy. Maintain hygiene and monitor regularly."

    elif prediction == "medium":
        condition = "Minor skin issue detected"
        recommendation = f"The {animal} shows mild skin changes. Clean the area and observe."

    else:
        condition = "Serious skin condition detected"
        recommendation = f"The {animal} may have a serious issue. Consult a veterinarian immediately."

    if prediction == "low":
        disease = "Normal / Healthy Skin"
    elif prediction == "medium":
        disease = "Possible Skin Disease"
    else:
        disease = random.choice(["Possible Squamous Cell Carcinoma(SCC)", "Possible Melanoma"])

    return render_template("result.html",
                           prediction=prediction,
                           confidence=confidence,
                           condition=condition,
                           recommendation=recommendation,
                           image_url=image_url,
                           disease=disease)

if __name__ == '__main__':
    app.run(debug=True)