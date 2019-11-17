from CRAFT_pytorch.test import CRAFTTextDetector
from services.image_service import ImageService
from deep_text_recognition_benchmark.demo import DeepTextRecognition
import cv2


class ImageTextDetector:
    def __init__ (self):
        self.craft_text_detector = CRAFTTextDetector()
        self.image_service = ImageService()
        self.deep_text_recognition = DeepTextRecognition()
    
    def get_phrases_from_image(self, image_path='books_images/book.jpg'):
        boxes, image = self.craft_text_detector.detect_one(image_path=image_path)
        phrases_images = self.image_service.get_phrases_from_image(boxes, image)
        # results = self.deep_text_recognition.get_text_from_image_box(image_arrays = phrases_images)
        return phrases_images


    def get_words_from_image(self, image_path='books_images/book.jpg'):
        boxes, image = self.craft_text_detector.detect_one(image_path=image_path)
        word_masked_images = self.image_service.get_words_from_image(boxes, image)
        results = self.deep_text_recognition.get_text_from_image_box(image_arrays = word_masked_images)
        return results, image
    
    def get_text(self):
        img = cv2.imread('deep_text_recognition_benchmark/demo_image/demo_1.png')
        image_arrays = []
        image_arrays.append(img)
        results = self.deep_text_recognition.get_text_from_image_box(image_arrays = image_arrays)
        return results
