import torch
from core.model_handler.BaseModelHandler import BaseModelHandler

class MagFaceModelHandler(BaseModelHandler):
    def __init__(self, model, device, cfg):
        super().__init__(model, device, cfg)
        self.model.to(device)
        self.model.eval()

    def inference_on_image(self, image):
        with torch.no_grad():
            img = self._preprocess(image)
            embedding = self.model(img.to(self.device))
        return embedding.cpu().numpy().flatten()
