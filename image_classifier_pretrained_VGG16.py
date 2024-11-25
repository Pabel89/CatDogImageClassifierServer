"""
Dieses Modul ist ein Bild Klassifizierer fuer Katzen und Hunde.
Die Genauigkeit liegt hier bei 97%, da Module wie das Feature Modul vom vortrainierten VGG16 Model 
von Pytorch genutzt wird.

Author: Alexander Pabel
"""
from PIL import Image
import torch
from torchvision import transforms,models
import torch.nn as nn
import torch.nn.functional as F

class CatDogClassifierCNNwithVGG16(nn.Module):
    def __init__(self):
        super(CatDogClassifierCNNwithVGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        # Einfrieren der konvolutionalen Layer damit das vorherige Wissen ueber die Features erhalten bleibt
        for param in self.vgg16.parameters():
            param.requires_grad = False
        #Ueberschreiben des Klassifierzungsmoduls des VGG16 Models um das Klassifizierungsmodul meines CNNs zu uebernehmen
        self.vgg16.classifier = nn.Sequential(nn.Flatten(),
            nn.Linear(25088, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),)
    def forward(self, x):
        x = self.vgg16(x)
        return x

def classifyImage(pathparam):
    print("classifyImage from VGG16 executed")
    model = CatDogClassifierCNNwithVGG16()
    model.load_state_dict(torch.load('model_state/model_VGG16.pth',weights_only=True),strict=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Bild laden
    image_path = pathparam
    image = Image.open(image_path).convert('RGB')

    # Anwenden der transformationen
    image = transform(image)
    
    # Stapel dimension angeben
    image = image.unsqueeze(0)  # Füge eine Stapeldimension hinzu
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image = image.to(device)
    
    # Modelvorhersagen bekommen
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Den Index der Vorhersage auf das Klassenlabel mappen
    idx_to_class = {0: 'cats', 1: 'dogs'}
    predicted_class = idx_to_class[predicted.item()]
    print(f'Predicted class: {predicted_class}')
    return predicted_class