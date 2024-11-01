import nlp2

from SoundCodec.base_codec.funcodec import BaseCodec
import os

class Codec(BaseCodec):
    def config(self):
        self.setting = "funcodec_en_libritts-16k-gr1nq32ds320"
        self.sampling_rate = 16000
        self.config_path = f"funcodec/{self.setting}/config.yaml"
        if not os.path.exists(self.config_path):
            nlp2.download_file(
                'https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch/raw/main/config.yaml',
                f"funcodec/{self.setting}")
            
        self.ckpt_path = f"funcodec/{self.setting}/model.pth"
        if not os.path.exists(self.ckpt_path):
            nlp2.download_file(
                'https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch/resolve/main/model.pth',
                f"funcodec/{self.setting}")
        

