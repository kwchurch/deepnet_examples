import sys,argparse
import soundfile as sf
import numpy as np
from espnet2.bin.tts_inference import Text2Speech

assert False, 'under construction'

# example is borrowed from https://www.paddlepaddle.org.cn/hubdetail?name=fastspeech_ljspeech&en_category=TextToSpeech
# example usage: python text_to_speech.py -o audio < sample_input.txt

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help='prefix for output wav files', default='audio')
args = parser.parse_args()

model = Text2Speech.from_pretrained("julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train")

for line in sys.stdin:
    rline = line.rstrip()
    if len(rline) > 0:
        speech,sampling_rate = model(rline)
        fn = f"{i}.wav"
        print(fn)
        sf.write(fn, speech, sampling_rate)

# @inproceedings{watanabe2018espnet,
#   author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
#   title={{ESPnet}: End-to-End Speech Processing Toolkit},
#   year={2018},
#   booktitle={Proceedings of Interspeech},
#   pages={2207--2211},
#   doi={10.21437/Interspeech.2018-1456},
#   url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
# }
# @inproceedings{hayashi2020espnet,
#   title={{Espnet-TTS}: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit},
#   author={Hayashi, Tomoki and Yamamoto, Ryuichi and Inoue, Katsuki and Yoshimura, Takenori and Watanabe, Shinji and Toda, Tomoki and Takeda, Kazuya and Zhang, Yu and Tan, Xu},
#   booktitle={Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
#   pages={7654--7658},
#   year={2020},
#   organization={IEEE}
# }
