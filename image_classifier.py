"""
Dieses Modul ist ein Bild Klassifizierer fuer Katzen und Hunde.
Zurzeit liegt die Genauigkeit des trainierten catDogClassifierCNNs bei 92%. 

Author: Alexander Pabel
"""
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

class catDogClassifierCNN(nn.Module):
    def __init__(self):
        super(catDogClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Dropout-Schichten zur Regularisierung nach den kovolutionalen Schichten um Overfitting zu vermeiden
        self.dropout_conv1 = nn.Dropout(p=0.2)  # Dropout nach der ersten konvolutionalen Schicht
        self.dropout_conv2 = nn.Dropout(p=0.3)  # Dropout nach der zweiten konvolutionalen Schicht
        self.dropout_conv3 = nn.Dropout(p=0.3)  # Dropout nach der dritten konvolutionalen Schicht
        
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout1 = nn.Dropout(p=0.5)  # Erhöhung der dropout rate nach der vollverbundenen Schicht
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout_conv1(x)  # Anwenden des dropouts nach der ersten konvolutionalen Schicht
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_conv2(x)  # Anwenden des dropouts nach der zweiten konvolutionalen Schicht
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout_conv3(x)  # Anwenden des dropouts nach der dritten konvolutionalen Schicht
        
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten: den mehrdimensionalen Output der konvolutionalen Schicht in ein eindimensionales Array
        # bringen für die vollverbundene Schicht
        x = x.view(-1, 256 * 8 * 8)
        
        # Fully connected layer with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = self.fc2(x)
        return x

def classifyImage(pathparam):
    model = catDogClassifierCNN()
    model.load_state_dict(torch.load('model_state/model.pth',weights_only=True),strict=True)
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