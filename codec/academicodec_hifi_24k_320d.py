import json
import nlp2
from base_codec.academicodec import BaseCodec


class Codec(BaseCodec):
    def config(self):
        self.setting = "academicodec_hifi_24k_320d"
        nlp2.download_file(
            'https://raw.githubusercontent.com/yangdongchao/AcademiCodec/master/egs/HiFi-Codec-24k-320d/config_24k_320d.json',
            'academicodec_hifi')
        self.config_path = "academicodec_hifi/config_24k_320d.json"
        nlp2.download_file(
            'https://huggingface.co/Dongchao/AcademiCodec/resolve/main/HiFi-Codec-24k-320d',
            "academicodec_hifi")
        self.ckpt_path = "academicodec_hifi/HiFi-Codec-24k-320d"

        with open(self.config_path, 'r') as f:
            config = json.load(f)
            self.sampling_rate = config['sampling_rate']
