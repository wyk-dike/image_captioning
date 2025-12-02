import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from net import ImageCaptioningModel
from dataset_utils import (
    get_train_val_json_data,
    build_vocabulary,
    TrainDataset_for_net
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")


class Config:
    train_json_path = r"E:\Dataset_for_machine_learning\Image_Captioning\annotations\train.json"
    val_json_path = r"E:\Dataset_for_machine_learning\Image_Captioning\annotations\val.json"
    train_image_dir = r"E:\Dataset_for_machine_learning\Image_Captioning\train"
    val_image_dir = r"E:\Dataset_for_machine_learning\Image_Captioning\val"

    vocab_size = 10000
    d_model = 512
    nhead = 8
    num_decoder_layers = 6
    dim_feedforward = 2048
    max_seq_len = 50
    dropout = 0.1

    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-5

    print_every = 100
    target_size = (224, 224)

config = Config()


def create_data_loaders():
    train_data = get_train_val_json_data(config.train_json_path)

    val_data = get_train_val_json_data(config.val_json_path)

    vocab = build_vocabulary(train_data, min_word_freq=2)

    config.vocab_size = len(vocab)

    train_dataset = TrainDataset_for_net(
        train_data=train_data,
        image_dir_path=config.train_image_dir,
        vocab=vocab,
        max_caption_length=config.max_seq_len,
        target_size=config.target_size
    )

    val_dataset = TrainDataset_for_net(
        train_data=val_data,
        image_dir_path=config.val_image_dir,
        vocab=vocab,
        max_caption_length=config.max_seq_len,
        target_size=config.target_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    print(f"Количество образцов обучающих наборов: {len(train_dataset)}, Размер выборки валидационного набора: {len(val_dataset)}")

    return train_loader, val_loader, vocab


def create_model():
    model = ImageCaptioningModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Общее количество параметров модели: {total_params:,}, Количество параметров, которые можно обучить: {trainable_params:,}")

    return model


def create_criterion_and_optimizer(model):
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=False
    )

    return criterion, optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch:02d} [Train]')

    for step, batch in enumerate(pbar):
        if batch is None:
            continue

        images = batch['image'].to(device)
        caption_tokens = batch['caption_tokens'].to(device)
        image_masks = batch['image_mask'].to(device)

        optimizer.zero_grad()
        outputs = model(images, caption_tokens, image_masks)

        targets = caption_tokens[:, 1:config.max_seq_len]

        outputs = outputs.reshape(-1, outputs.size(-1))
        targets = targets.reshape(-1)

        loss = criterion(outputs, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss / total_samples:.4f}'
        })

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    return avg_loss


def validate_epoch(model, val_loader, criterion, epoch):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch:02d} [Val]')

        for batch in pbar:
            if batch is None:
                continue

            images = batch['image'].to(device)
            caption_tokens = batch['caption_tokens'].to(device)
            image_masks = batch['image_mask'].to(device)

            outputs = model(images, caption_tokens, image_masks)

            targets = caption_tokens[:, 1:config.max_seq_len]

            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / total_samples:.4f}'
            })

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    return avg_loss


def main():
    start_time = time.time()

    start_epoch = 1

    train_loader, val_loader, vocab = create_data_loaders()

    model = create_model()

    criterion, optimizer, scheduler = create_criterion_and_optimizer(model)

    for epoch in range(start_epoch, config.num_epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'=' * 50}")
        time.sleep(0.5)

        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)

        val_loss = validate_epoch(model, val_loader, criterion, epoch)

        scheduler.step(val_loss)

        print(f"Epoch {epoch} : Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    model_infos = {
        'model_state_dict': model.state_dict(),

        'vocab': {
            'word2idx': vocab.word2idx,
            'idx2word': vocab.idx2word,
            'special_tokens': {
                '<pad>': vocab['<pad>'],
                '<start>': vocab['<start>'],
                '<end>': vocab['<end>'],
                '<unk>': vocab['<unk>']
            }
        },

        'config': {
            'vocab_size': config.vocab_size,
            'd_model': config.d_model,
            'nhead': config.nhead,
            'num_decoder_layers': config.num_decoder_layers,
            'dim_feedforward': config.dim_feedforward,
            'max_seq_len': config.max_seq_len,
            'dropout': config.dropout,
            'target_size': config.target_size
        }
    }

    torch.save(model_infos, "ImageCaptioningModel.pth")

    training_time = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"Завершение обучения!")
    print(f"Модель сохраняется в: ./ImageCaptioningModel.pth")
    print(f"Общее время обучения: {training_time / 60:.2f} min")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()

