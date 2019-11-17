
class ConfigureCRAFTPytorch:
    def __init__(self, trained_model, test_folder):
        self.trained_model = trained_model
        self.test_folder = test_folder
        self.text_threshold = 0.7
        self.low_text = 0.4
        self.link_threshold = 0.4
        self.cuda = False
        self.canvas_size = 1280
        self.mag_ratio = 1.5
        self.poly = False
        self.show_time = False
        self.refine = False
        self.refiner_model = 'weights/craft_refiner_CTW1500.pth'