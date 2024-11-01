import nlp2

from SoundCodec.base_codec.funcodec import BaseCodec
import os

class Codec(BaseCodec):
    def config(self):
        self.setting = "funcodec_zh_en_general_16k_nq32ds640"
        self.sampling_rate = 16000
        self.config_path = f"funcodec/{self.setting}/config.yaml"
        if not os.path.exists(self.config_path):
            nlp2.download_file(
                'https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/raw/main/config.yaml',
                f"funcodec/{self.setting}")
        self.ckpt_path = f"funcodec/{self.setting}/model.pth"
        if not os.path.exists(self.ckpt_path):
            nlp2.download_file(
                'https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/resolve/main/model.pth',
                f"funcodec/{self.setting}")
        
