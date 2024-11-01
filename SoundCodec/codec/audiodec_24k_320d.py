from SoundCodec.base_codec.audiodec import BaseCodec
import nlp2
import os

class Codec(BaseCodec):
    def config(self):
        self.setting = "audiodec_24k_320d"
        try:
            from AudioDec.utils.audiodec import AudioDec as AudioDecModel, assign_model
        except:
            raise Exception("Please install AudioDec first. pip install git+https://github.com/voidful/AudioDec.git")
        
        # # download encoder
        # nlp2.download_file(
        #     'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/autoencoder/symAD_libritts_24000_hop300/checkpoint-500000steps.pkl',
        #     'audiodec_autoencoder_24k_320d')
        # nlp2.download_file(
        #     'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/autoencoder/symAD_libritts_24000_hop300/config.yml',
        #     "audiodec_autoencoder_24k_320d")
        
        # self.encoder_config_path = "audiodec_autoencoder_24k_320d/checkpoint-500000steps.pkl"
        
        self.encoder_config_path = "audiodec/autoencoder/symAD_libritts_24000_hop300/checkpoint-1000000steps.pkl"
        if not os.path.exists(self.encoder_config_path):
            raise Exception("encoder_config_path {encoder_config_path} not found, Please download the model from huggingface first.")
        
        # download decoder
        # nlp2.download_file(
        #     'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/checkpoint-500000steps.pkl',
        #     'audiodec_vocoder_24k_320d')
        # nlp2.download_file(
        #     'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/config.yml',
        #     "audiodec_vocoder_24k_320d")
        # nlp2.download_file(
        #     "https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/symAD_libritts_24000_hop300_clean.npy",
        #     "audiodec_vocoder_24k_320d"
        # )
        # self.decoder_config_path = "audiodec_vocoder_24k_320d/checkpoint-500000steps.pkl"
        
        self.decoder_config_path = "audiodec/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/checkpoint-500000steps.pkl"
        if not os.path.exists(self.decoder_config_path):
            raise Exception("decoder_config_path {decoder_config_path} not found, Please download the model from huggingface first.")
        
        self.sampling_rate = 24000
        audiodec_model = AudioDecModel(tx_device=self.device, rx_device=self.device)
        audiodec_model.load_transmitter(self.encoder_config_path)
        audiodec_model.load_receiver(self.encoder_config_path, self.decoder_config_path)
        self.model = audiodec_model
