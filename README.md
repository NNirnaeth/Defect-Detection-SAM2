PROJECT STRUCTURE

sam2/
├── src/
│   ├── main.py                  # Entry point (training / evaluation)
│   ├── train.py                 # Training logic
│   ├── evaluate.py             # Single image + test set evaluation
│   ├── utils.py                # Batch reading, decoding, IoU, etc.
│   └── dataloaders/
│       ├── dataset_gc10.py     # Loader for GC10-DET
│       ├── dataset_kolektor.py # Loader for KolektorSDD2
│       └── dataset_chest.py    # Loader for Chest X-ray
├── datasets/                   # All datasets go here
├── models/                     # Checkpoints (SAM2 base and fine-tuned)
├── configs/                    # SAM2 config YAML files
└── README.md

DATASET FORMAT

GC10-DET
datasets/gc10-DET/
└── ds/
    ├── img/*.jpg
    └── ann/*.json

KolektorSDD2

datasets/kolektorSDD2/
├── train/*.jpg + *.json
├── test/*.jpg + *.json
└── val/*.jpg + *.json

Chest X-rayChest

datasets/chest/
├── train.csv          # Contains ImageId and MaskId columns
├── images/images/*.jpg
└── masks/masks/*.png

INSTALLATION

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

USAGE

Train SAM2 on GC10

python src/main.py train \
    --dataset gc10 \
    --checkpoint models/sam2_hiera_small.pt \
    --config configs/sam2/sam2_hiera_s.yaml

Evaluate one image(random)

python src/main.py eval_one_image \
    --dataset gc10 \
    --checkpoint models/sam2_hiera_small.pt \
    --config configs/sam2/sam2_hiera_s.yaml \
    --fine_tuned_checkpoint models/fine_tuned_sam2_1000.torch

Evaluate full test set (mean IoU)
python src/main.py eval_full_test \
    --dataset gc10 \
    --checkpoint models/sam2_hiera_small.pt \
    --config configs/sam2/sam2_hiera_s.yaml \
    --fine_tuned_checkpoint models/fine_tuned_sam2_1000.torch

NOTES

Default image size is resized to max 1024px (maintaining aspect ratio).

Supports RLE-encoded masks (.json) or direct binary masks (.png).

Fine-tuned weights are saved every 500 steps.

CONTACT

Natalia Peinado Verhoeven - TFM UCJC 2025
natalia.peinado@alumno.ucjc.edu
