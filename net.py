"""
CNN + Transformer

ResNet50 Encoder
Transformer Decoder
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


# ================================== Кодирование позиций ==================================
class NoLearnPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=50):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# ================================== Кодировщик изображений ==================================
class ImageEncoder(nn.Module):
    def __init__(self, d_model=512, max_seq_len=50):
        super().__init__()

        resnet50 = models.resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])

        self.feature_projection = nn.Linear(
            in_features=2048,
            out_features=d_model
        )

        self.position_encoding = NoLearnPositionalEncoding(d_model, max_seq_len=max_seq_len)

    def forward(self, x, padding_mask=None):
        features = self.backbone(x)

        batch_size, channels, height, width = features.shape
        features = features.view(batch_size, channels, -1)
        features = features.permute(0, 2, 1)

        features = self.feature_projection(features)

        position_features = self.position_encoding(features)

        feature_mask = None
        if padding_mask is not None:
            feature_mask = F.interpolate(
                padding_mask.unsqueeze(1).float(),
                size=(height, width),
                mode='nearest'
            ).squeeze(1)

            feature_mask = feature_mask.view(batch_size, -1)

            feature_mask = feature_mask.bool()

        return position_features, feature_mask


# ================================== Вложение текста ==================================
class TextEmbedder(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len=50, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0
        )

        self.position_encoding = NoLearnPositionalEncoding(d_model, max_seq_len=max_seq_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        token_embedded = self.token_embedding(text)

        token_embedded = token_embedded * math.sqrt(self.d_model)

        embedded_text = self.position_encoding(token_embedded)

        embedded_text = self.dropout(embedded_text)

        return embedded_text


# ================================== Модель ==================================
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, max_seq_len=50, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.image_encoder = ImageEncoder(d_model, max_seq_len)

        self.text_embedder = TextEmbedder(vocab_size, d_model, max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_layer = nn.Linear(in_features=d_model, out_features=vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, images, caption_tokens, padding_mask=None):
        image_features, feature_mask = self.image_encoder(images, padding_mask)

        text_embedded = self.text_embedder(caption_tokens[:, :-1])

        tgt_seq_len = caption_tokens[:, :-1].size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(caption_tokens.device)

        decoder_output = self.transformer_decoder(
            tgt=text_embedded,
            memory=image_features,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=feature_mask
        )

        output = self.output_layer(decoder_output)

        return output

    def generate(self, images, start_token, end_token, max_len=50, padding_mask=None):
        batch_size = images.size(0)
        device = images.device

        image_features, feature_mask = self.image_encoder(images, padding_mask)

        input_seq = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        for i in range(max_len):
            text_embedded = self.text_embedder(input_seq)

            tgt_mask = self.generate_square_subsequent_mask(input_seq.size(1)).to(device)

            decoder_output = self.transformer_decoder(
                tgt=text_embedded,
                memory=image_features,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=feature_mask
            )

            next_word_logits = self.output_layer(decoder_output[:, -1, :])

            next_word = torch.argmax(next_word_logits, dim=-1, keepdim=True)

            input_seq = torch.cat([input_seq, next_word], dim=1)

            if (next_word == end_token).all():
                break

        return input_seq

