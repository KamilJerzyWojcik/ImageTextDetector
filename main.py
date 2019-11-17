from CRAFT_pytorch.test import CRAFTTextDetector
from services.image_service import ImageService
from deep_text_recognition_benchmark.demo import DeepTextRecognition

class ImageTextDetector:
    def __init__ (self):
        self.craft_text_detector = CRAFTTextDetector()
        self.image_service = ImageService()
        self.deep_text_recognition = DeepTextRecognition()
    
    def get_text_from_image(self, image_path='books_images/book.jpg'):
         boxes, image = self.craft_text_detector.detect_one(image_path=image_path)
         text_images = self.image_service.get_words_from_image(boxes, image)
         return text_images
    
    def get_text(self):
        self.deep_text_recognition.get_text_from_image_box()
