# Tacotron2

Implementation of text-to-speech model Tacotron2.

## Prepare data

Before using download LJSpeech dataset:

```
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xf LJSpeech-1.1.tar.bz2
```

and pretrained vocoder:

```python
from google_drive_downloader import GoogleDriveDownloader as gdd
gdd.download_file_from_google_drive(
    file_id='1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF',
    dest_path='./waveglow_256channels_universal_v5.pt'
)
```
```
git clone https://github.com/NVIDIA/waveglow.git
```

To setup docker container run ./build_image.sh and ./run_container.sh

## Train

run "train.py number_of_epochs", for example "train.py 10"

Data will be logged in wandb.

## Inference

run "inference.py "text to generate""

Wav file will be saved in gen.wav as well as in wandb.
