import torch
import cv2
import os


import warnings
warnings.filterwarnings("ignore")

from net import ImageCaptioningModel
from dataset_utils import OpenCVResizer, image_transforms, mask_to_tensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")


class ImageCaptioningInference:
    def __init__(self, model_path):
        model_infos = torch.load(model_path, map_location=device)

        self.vocab = model_infos['vocab']
        self.idx2word = self.vocab['idx2word']
        self.word2idx = self.vocab['word2idx']

        self.config = model_infos['config']
        print(f"Конфигурация модели: {self.config}")

        self.model = ImageCaptioningModel(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_decoder_layers=self.config['num_decoder_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            max_seq_len=self.config['max_seq_len'],
            dropout=self.config['dropout']
        )

        self.model.load_state_dict(model_infos['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        self.image_transforms = image_transforms

        self.resizer = OpenCVResizer(target_size=self.config['target_size'])

        self.start_token = self.word2idx['<start>']
        self.end_token = self.word2idx['<end>']
        self.pad_token = self.word2idx['<pad>']

    def preprocess_image(self, image_path):
        try:
            image_pil, mask_pil, _ = self.resizer(image_path)

            image_tensor = self.image_transforms(image_pil)  # [3, H, W]

            image_mask = mask_to_tensor(mask_pil)  # [H, W]

            image_tensor = image_tensor.unsqueeze(0)  # [1, 3, H, W]
            image_mask = image_mask.unsqueeze(0)  # [1, H, W]

            return image_tensor, image_mask

        except Exception as e:
            print(e)
            return None, None

    def tokens_to_sentence(self, tokens):
        words = []
        for token in tokens:
            if token == self.end_token:
                break
            if token not in [self.start_token, self.pad_token]:
                word = self.idx2word.get(token, '<unk>')
                words.append(word)

        return ' '.join(words)

    def generate_caption(self, image_path, max_length=50):
        print(f"\nГенерация описаний для изображений: {os.path.basename(image_path)}")

        image_tensor, image_mask = self.preprocess_image(image_path)
        if image_tensor is None:
            return None

        image_tensor = image_tensor.to(device)
        image_mask = image_mask.to(device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                images=image_tensor,
                start_token=self.start_token,
                end_token=self.end_token,
                max_len=max_length,
                padding_mask=image_mask
            )

        caption = self.tokens_to_sentence(generated_tokens[0].cpu().numpy())

        return caption


def main():
    model_path = "ImageCaptioningModel.pth"

    inference = ImageCaptioningInference(model_path)

    while True:
        try:
            image_path = input("Введите путь к изображению: ").strip()

            if image_path == "exit":
                break

            caption = inference.generate_caption(image_path)
            print(f"\nСгенерированное описание: {caption}")

            image = cv2.imread(image_path)
            h, w, c = image.shape
            scale_w = 500 / w
            scale_h = 500 / h
            scale = min(scale_w, scale_h)
            resized_image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
            cv2.imshow("Image", resized_image)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()


