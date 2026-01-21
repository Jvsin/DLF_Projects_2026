import torch
import torch.nn as nn
import torchvision.models as models

class CrossModalNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim, pretrained_embeddings=None):
        super(CrossModalNetwork, self).__init__()
        
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        
        modules = list(resnet.children())[:-1] 
        self.resnet_base = nn.Sequential(*modules)
        
        for param in self.resnet_base.parameters():
            param.requires_grad = False

        for param in self.resnet_base[7].parameters():
            param.requires_grad = True
            
        self.image_projection = nn.Sequential(
            nn.Linear(2048, visual_dim),
            nn.BatchNorm1d(visual_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True 
            
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.text_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, visual_dim),
            nn.BatchNorm1d(visual_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, images, captions):
        img_features = self.resnet_base(images).squeeze(-1).squeeze(-1)
        img_emb = self.image_projection(img_features)
        
        embeds = self.embedding(captions)
        lstm_out, _ = self.lstm(embeds)
        txt_features = torch.mean(lstm_out, dim=1) 
        txt_emb = self.text_projection(txt_features)
        
        fused_vector = img_emb * txt_emb
        
        output = self.classifier(fused_vector)
        return output