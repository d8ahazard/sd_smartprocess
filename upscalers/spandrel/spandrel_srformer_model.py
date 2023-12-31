import os.path

from extensions.sd_smartprocess.upscalers.spandrel.spandrel_upscaler_base import SpandrelUpscaler
from modules import modelloader
from modules.upscaler import UpscalerData

model_urls = [
    "https://github.com/d8ahazard/sd_smartprocess/releases/download/1.0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth",

]

models = {
    "SRformer 4xFF": "https://github.com/d8ahazard/sd_smartprocess/releases/download/1.0.0/4xFrankendata_FullDegradation_g.pth",
    "SRFormer 4xNomos8kSC": "https://github.com/d8ahazard/sd_smartprocess/releases/download/1.0.0/4xNomos8kSC_SRFormer.pth",
    "SRFormer FP": "https://github.com/d8ahazard/sd_smartprocess/releases/download/1.0.0/FrankendataPretrainer_SRFormer_g.pth",
    "SwinIR x8": "https://github.com/d8ahazard/sd_smartprocess/releases/download/1.0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth",
    "4xLDSIR": "https://github.com/d8ahazard/sd_smartprocess/releases/download/1.0.0/4xLSDIR_DAT.pth",
    "ClearRealityV1 4x": "https://github.com/d8ahazard/sd_smartprocess/releases/download/1.0.0/4x-ClearRealityV1_SPAN.pth"
}


class SpandrelSRFormerModel(SpandrelUpscaler):
    scale = 4
    name = "SRFormer"

    def __init__(self, create_dirs=False):
        super().__init__(create_dirs)
        self.name = "SRFormer"
        self.scale = 4
        user_models = self.find_models(ext_filter=[".pth"])
        self.scalers = []
        added_models = []
        for file in user_models:
            model_name = os.path.basename(file)
            display_name = modelloader.friendly_name(file)
            for pre_name, model_url in models.items():
                if model_name in model_url:
                    display_name = pre_name
                    self.scalers.append(UpscalerData(display_name, file, self))
                    added_models.append(display_name)
                    break
            if display_name not in added_models:
                self.scalers.append(UpscalerData(display_name, file, self))
                added_models.append(display_name)
        for model_name, model_url in models.items():
            if model_name not in added_models:
                file_name = os.path.basename(model_url)
                model_path = modelloader.load_file_from_url(model_url, model_dir=self.model_path, file_name=file_name)
                self.scalers.append(UpscalerData(model_name, model_path, self))
