import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import os
from PIL import Image

from .utils import CTCLabelConverter, AttnLabelConverter
from .dataset import RawDataset, AlignCollate, RawDatasetArray
from .model import Model
from .configure_text_recognition import ConfigureTextRecognition



class DeepTextRecognition:

    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        self.configure_text_recognition = ConfigureTextRecognition(
            saved_model= '../neural_networks/TextRecognition/TPS-ResNet-BiLSTM-Attn.pth'
        )
        if self.configure_text_recognition.sensitive:
            self.configure_text_recognition.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        if self.configure_text_recognition.rgb:
                self.configure_text_recognition.input_channel = 3
        cudnn.benchmark = True
        cudnn.deterministic = True
        self.configure_text_recognition.num_gpu = torch.cuda.device_count()


    def get_text_from_image_box(self, image_arrays):
            """
                change configuration if use another nn
            """
            converter = self.get_converter()

            model = Model(self.configure_text_recognition)
            model = torch.nn.DataParallel(model).to(device)
            print('loading pretrained model from %s' % self.configure_text_recognition.saved_model)
            model.load_state_dict(torch.load(self.configure_text_recognition.saved_model, map_location=device))

            demo_loader = self.get_demo_loader(image_arrays)

            # predict
            model.eval()
            results = []
            with torch.no_grad():
                for image_tensors, image_name in demo_loader:
                    batch_size = image_tensors.size(0)
                    image = image_tensors.to(device)
                    length_for_pred = torch.IntTensor([self.configure_text_recognition.batch_max_length] * batch_size).to(device)
                    text_for_pred = torch.LongTensor(batch_size, self.configure_text_recognition.batch_max_length + 1).fill_(0).to(device)

                    if 'CTC' in self.configure_text_recognition.Prediction:
                        preds = model(image, text_for_pred).log_softmax(2)
                        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                        _, preds_index = preds.max(2)
                        preds_index = preds_index.view(-1)
                        preds_str = converter.decode(preds_index.data, preds_size.data)

                    else:
                        preds = model(image, text_for_pred, is_train=False)
                        _, preds_index = preds.max(2)
                        preds_str = converter.decode(preds_index, length_for_pred)

                    print('-' * 80)
                    print(f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score')
                    print('-' * 80)
                    preds_prob = F.softmax(preds, dim=2)
                    preds_max_prob, _ = preds_prob.max(dim=2)
                    for img_name, pred, pred_max_prob in zip(image_name, preds_str, preds_max_prob):
                        if 'Attn' in self.configure_text_recognition.Prediction:
                            pred_EOS = pred.find('[s]')
                            pred = pred[:pred_EOS]
                            pred_max_prob = pred_max_prob[:pred_EOS]
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                        if confidence_score > self.configure_text_recognition.min_confidence_score:
                            results.append(pred)
                        
                        print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
            return results


    def get_converter(self):
        if 'CTC' in self.configure_text_recognition.Prediction:
            converter = CTCLabelConverter(self.configure_text_recognition.character)
        else:
             converter = AttnLabelConverter(self.configure_text_recognition.character)
                
        self.configure_text_recognition.num_class = len(converter.character)
        return converter


    def get_demo_loader(self, image_arrays):
        AlignCollate_demo = AlignCollate(
            imgH=self.configure_text_recognition.imgH, 
            imgW=self.configure_text_recognition.imgW, 
            keep_ratio_with_pad=self.configure_text_recognition.PAD
        )
        demo_data = RawDatasetArray(img_arrays = image_arrays, opt=self.configure_text_recognition)
        return torch.utils.data.DataLoader(
            demo_data, batch_size=self.configure_text_recognition.batch_size,
            shuffle=False,
            num_workers=int(self.configure_text_recognition.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)


    def get_text_from_images(self):
        """ model configuration """
        if 'CTC' in self.configure_text_recognition.Prediction:
            converter = CTCLabelConverter(self.configure_text_recognition.character)
        else:
            converter = AttnLabelConverter(self.configure_text_recognition.character)
            
        self.configure_text_recognition.num_class = len(converter.character)

        if self.configure_text_recognition.rgb:
            self.configure_text_recognition.input_channel = 3
        
        model = Model(self.configure_text_recognition)
        model = torch.nn.DataParallel(model).to(device)

        # load model
        print('loading pretrained model from %s' % self.configure_text_recognition.saved_model)
        torch_load = torch.load(self.configure_text_recognition.saved_model, map_location=device)
        model.load_state_dict(torch_load)

        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(imgH=self.configure_text_recognition.imgH, imgW=self.configure_text_recognition.imgW, keep_ratio_with_pad=self.configure_text_recognition.PAD)
        demo_data = RawDataset(root=self.configure_text_recognition.image_folder, opt=self.configure_text_recognition)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=self.configure_text_recognition.batch_size,
            shuffle=False,
            num_workers=int(self.configure_text_recognition.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)

        # predict
        model.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([self.configure_text_recognition.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, self.configure_text_recognition.batch_max_length + 1).fill_(0).to(device)

                if 'CTC' in self.configure_text_recognition.Prediction:
                    preds = model(image, text_for_pred).log_softmax(2)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    preds_index = preds_index.view(-1)
                    preds_str = converter.decode(preds_index.data, preds_size.data)

                else:
                    preds = model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

                print('-' * 80)
                print(f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score')
                print('-' * 80)
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in self.configure_text_recognition.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                    # print(f'{img_name}\t{pred}\t{confidence_score:0.4f}')
                    print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')

