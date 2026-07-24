# Plant Disease Prediction

PyTorch pipeline for plant leaf disease classification using AlexNet with transfer learning on the PlantVillage dataset.

## Dataset

PlantVillage in ImageFolder layout:

```
dataset/PlantVillage/
├── train/
└── val/
```

Each subfolder is a disease class (39 classes total).

## Model

AlexNet with a replaced final classifier layer. A pretrained checkpoint is provided at `models/alexnet.pkl`.

## Project Structure

```
PlantDiseasePrediction/
├── dataset/PlantVillage/
├── models/alexnet.pkl
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
├── results/
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

Training is not run automatically. From the `src` directory:

```bash
python train.py --epochs 30 --batch-size 32 --lr 1e-4
```

Optional transfer learning from the existing checkpoint:

```bash
python train.py --pretrained ../models/alexnet.pkl
```

Best weights are saved to `results/best_model.pth`.

## Evaluation

```bash
python evaluate.py --model-path ../results/best_model.pth
```

Reports accuracy, precision, recall, F1-score, and confusion matrix.

## Inference

Uses `models/alexnet.pkl` by default:

```bash
python inference.py --image path/to/leaf.jpg
```

Output:

```
Image: path/to/leaf.jpg
Prediction: Tomato___Early_blight
Confidence: 0.9521
```
