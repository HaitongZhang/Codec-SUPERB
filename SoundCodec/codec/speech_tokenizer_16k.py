from SoundCodec.base_codec.speech_tokenizer import BaseCodec
import nlp2
import os

class Codec(BaseCodec):
    def config(self):
        self.config_path = "SpeechTokenizer/speechtokenizer_hubert_avg/config.json"
        # self.config_path = "speechtokenizer_hubert_avg/config.json"
        if not os.path.exists(self.config_path):
            nlp2.download_file(
                'https://huggingface.co/fnlp/SpeechTokenizer/raw/main/speechtokenizer_hubert_avg/config.json',
                'speechtokenizer_hubert_avg')
        self.ckpt_path = "SpeechTokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt"
        # self.ckpt_path = "speechtokenizer_hubert_avg/SpeechTokenizer.pt"
        if not os.path.exists(self.ckpt_path):
            nlp2.download_file(
                'https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt',
                "speechtokenizer_hubert_avg")
        
