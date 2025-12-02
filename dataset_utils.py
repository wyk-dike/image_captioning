import os
import cv2
import torch
import numpy as np
import json
from PIL import Image
from torchvision import transforms


class OpenCVResizer:
    def __init__(self, target_size=(1024, 1024)):
        self.t_w, self.t_h = target_size

    def __call__(self, image_path):
        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        scale_w = self.t_w / w
        scale_h = self.t_h / h
        scale = min(scale_w, scale_h)

        if scale >= 1:
            new_w = w
            new_h = h
            resized_image = image
        else:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        new_image = np.zeros((self.t_h, self.t_w, 3), dtype=np.uint8)
        mask = np.ones((self.t_h, self.t_w), dtype=np.uint8) * 255

        new_image[:new_h, :new_w] = resized_image
        mask[:new_h, :new_w] = 0

        new_image = Image.fromarray(new_image)
        mask = Image.fromarray(mask)
        scale_image_size = (new_w, new_h)

        return new_image, mask, scale_image_size


image_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB") if x.mode != 'RGB' else x),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def mask_to_tensor(mask):
    mask_np = torch.from_numpy(np.array(mask))
    mask_tensor = (mask_np > 0).float()
    return mask_tensor


def get_json_data(json_file_path):
    with open(json_file_path, 'r') as file:
        file_data = json.load(file)
    return file_data


def get_train_val_json_data(json_file_path):
    all_data = get_json_data(json_file_path)

    images = all_data['images']
    annotations = all_data['annotations']

    annotations = [
        ann
        for ann in annotations
        if ann['is_precanned'] == False and ann['is_rejected'] == False
    ]

    image_dict = {img['id']: img for img in images}

    result = []

    for ann in annotations:
        image_id = ann['image_id']

        image = image_dict[image_id]

        result.append({
            'image_id': image['id'],
            'image_name': image['file_name'],
            'annotation_id': ann['id'],
            'caption': ann['caption']
        })

    return result


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.word2idx.get(key, self.word2idx['<unk>'])
        elif isinstance(key, int):
            return self.idx2word.get(key, '<unk>')


def build_vocabulary(train_data, min_word_freq=2):
    word_freq = {}
    for item in train_data:
        caption = item['caption'].lower().strip()
        words = caption.split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

    vocab = Vocabulary()

    for word, freq in word_freq.items():
        if freq >= min_word_freq:
            vocab.add_word(word)

    return vocab


class TrainDataset_for_net(torch.utils.data.Dataset):
    def __init__(self, train_data, image_dir_path, vocab, max_caption_length=50, target_size=(224, 224)):
        super().__init__()

        self.train_data = train_data
        self.image_dir = image_dir_path
        self.vocab = vocab
        self.max_caption_length = max_caption_length
        self.target_size = target_size

        self.resizer = OpenCVResizer(target_size=target_size)

    def preprocess_caption(self, caption):
        caption = caption.lower().strip()

        words = caption.split()

        tokens = [self.vocab['<start>']]

        for word in words:
            if word in self.vocab.word2idx:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab['<unk>'])

        tokens.append(self.vocab['<end>'])

        if len(tokens) > self.max_caption_length:
            tokens = tokens[:self.max_caption_length]
            tokens[-1] = self.vocab['<end>']
        else:
            tokens = tokens + [self.vocab['<pad>']] * (self.max_caption_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.train_data[idx]
        image_name = item['image_name']
        caption = item['caption']

        image_path = os.path.join(self.image_dir, image_name)

        try:
            image_pil, mask_pil, _ = self.resizer(image_path)

            image_tensor = image_transforms(image_pil)  # [3, H, W]

            image_mask = mask_to_tensor(mask_pil)  # [H, W]

            caption_tokens = self.preprocess_caption(caption)  # [max_caption_length]

            return {
                'image': image_tensor,
                'caption_tokens': caption_tokens,
                'image_mask': image_mask,
                'image_id': item['image_id'],
                'annotation_id': item['annotation_id']
            }

        except Exception as e:
            print(e)
            return None

    def __len__(self):
        return len(self.train_data)

