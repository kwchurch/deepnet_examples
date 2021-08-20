import sys,argparse
import paddlehub as hub
import soundfile as sf
import numpy as np

# example is borrowed from https://www.paddlepaddle.org.cn/hubdetail?name=fastspeech_ljspeech&en_category=TextToSpeech
# example usage: python text_to_speech.py -m transformer_tts_ljspeech -o audio < sample_input.txt

# assert False, 'under construction'

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_string", help='deepvoice3_ljspeech|fastspeech_ljspeech|transformer_tts_ljspeech', default='deepvoice3_ljspeech')
parser.add_argument("-o", "--output", help='prefix for output wav files', default='audio')
args = parser.parse_args()

# Load fastspeech_ljspeech module.
module = hub.Module(name=args.model_string)

test_texts = sys.stdin.read().split('\n')
wavs, sample_rate = module.synthesize(texts=test_texts)

for i,wav in enumerate(wavs):
    filename = f"{i}.wav"
    print(filename)
    sf.write(filename, wav, sample_rate)
