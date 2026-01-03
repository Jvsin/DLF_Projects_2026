import torch
import torch.nn as nn
from torchvision import models
from transformers import DistilBertModel

class VerificationModel(nn.Module):
    def __init__(self):
        super(VerificationModel, self).__init__()
        
        #resNET 18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Zmieniamy ostatnią warstwę (2048 -> 768)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 768)
        
        # DistilBERT
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        #resnet freeze
        for param in self.resnet.parameters():
            param.requires_grad = False
        #ostatnia warstwa (fc) odmrożona
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        # bert freeze
        for param in self.bert.parameters():
            param.requires_grad = False

        #fusion
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, images, input_ids, attention_mask):
        #image
        img_features = self.resnet(images) # [Batch, 768]
        
        #text
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.last_hidden_state[:, 0, :] # [Batch, 768] (Token CLS)
        
        #fusion (multiplication)
        combined_features = img_features * text_features
        
        #prediction
        output = self.classifier(combined_features)
        return output