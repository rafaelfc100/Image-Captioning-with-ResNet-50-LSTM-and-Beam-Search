# Image Captioning with ResNet-50, LSTM and Beam Search

This project implements an Image Captioning system capable of automatically generating natural language descriptions from images. The model combines Computer Vision and Natural Language Processing techniques through an Encoder-Decoder architecture that uses a pretrained ResNet-50 as visual encoder and a two-layer LSTM as text decoder.

Unlike the original Show and Tell architecture, this implementation injects visual information into every decoding step, allowing the model to maintain image context throughout the caption generation process. Additionally, Beam Search is used during inference to generate more coherent and descriptive captions.

## Project Objectives

* Generate natural language descriptions from images.
* Combine deep learning techniques from Computer Vision and NLP.
* Evaluate caption quality using BLEU and Perplexity metrics.
* Compare greedy decoding against Beam Search strategies.
* Deploy the model through an interactive web application.

## Dataset

The model was trained using the Flickr30k dataset, a large-scale image-caption dataset containing photographs paired with human-written descriptions.

### Dataset Split

| Subset     | Images | Captions |
| ---------- | ------ | -------- |
| Training   | 25,426 | 127,129  |
| Validation | 3,178  | 15,890   |
| Test       | 3,179  | 15,895   |

A controlled vocabulary was created by removing rare words and incorporating special tokens:

* `<PAD>`
* `<SOS>`
* `<EOS>`
* `<UNK>`

Final vocabulary size:

* 11,420 tokens

## Model Architecture

### Encoder: ResNet-50

The encoder uses a pretrained ResNet-50 model trained on ImageNet.

Features:

* Output feature vector: 2048 dimensions
* Batch Normalization
* Dropout Regularization
* Projection into embedding space
* Partial fine-tuning of the last residual block

### Decoder: Two-Layer LSTM

The decoder generates captions word by word using:

* Embedding dimension: 256
* Hidden size: 512
* 2 LSTM layers
* Dropout: 0.5

Unlike traditional implementations, the image embedding is concatenated with every word embedding during generation:

```text
Word Embedding + Visual Embedding
             ↓
            LSTM
             ↓
      Next Word Prediction
```

This strategy helps the model retain visual context throughout the sentence.

## Training Configuration

| Hyperparameter    | Value             |
| ----------------- | ----------------- |
| Optimizer         | Adam              |
| Learning Rate     | 4e-4              |
| Batch Size        | 128               |
| Weight Decay      | 1e-4              |
| Dropout           | 0.5               |
| Gradient Clipping | 5.0               |
| Scheduler         | ReduceLROnPlateau |
| Trained Epochs    | 20                |

### Training Techniques

* Teacher Forcing
* Packed Sequences
* Gradient Clipping
* Learning Rate Scheduling
* Checkpoint Saving

## Beam Search Decoding

Instead of selecting only the most probable word at each step (Greedy Search), the model uses Beam Search.

Configurations tested:

* Beam = 1 (Greedy)
* Beam = 3
* Beam = 5
* Beam = 7

Beam Search maintains multiple candidate sentences simultaneously and selects the sequence with the highest normalized probability score.

## Results

### Validation Performance

| Metric          | Value  |
| --------------- | ------ |
| Validation Loss | 2.9466 |
| Perplexity      | 19.03  |

### BLEU Scores (Beam = 5)

| Metric | Score  |
| ------ | ------ |
| BLEU-1 | 0.6386 |
| BLEU-2 | 0.4304 |
| BLEU-3 | 0.2988 |
| BLEU-4 | 0.2035 |

### Inference Speed

| Metric            | Value |
| ----------------- | ----- |
| Images per Second | 7.98  |

## Generalization Analysis

The model was evaluated on images that were not part of Flickr30k.

### Strengths

* Correct identification of main subjects.
* Accurate recognition of common actions.
* Grammatically correct sentences.
* Consistent sentence structure.

### Limitations

* Occasional hallucinations.
* Confusion between visually similar scenes.
* Generic descriptions with limited detail.
* Lack of spatial attention mechanisms.

Examples of generated captions include:

> "a group of construction workers working on a train"

> "two construction workers are working on the side of a building"

These examples demonstrate that the model successfully captures the main semantic content of unseen images.

## Web Application

A Flask-based web interface was developed to allow real-time caption generation.

### Features

* Image upload by drag-and-drop or file selection.
* Multiple decoding modes.
* Inference time measurement.
* CPU/GPU execution information.
* Testing on external images.

### Available Decoding Modes

* Greedy Search
* Beam Search (3)
* Beam Search (5)
* Beam Search (7)

## Technologies

* Python
* PyTorch
* Torchvision
* ResNet-50
* LSTM
* Flask
* NLTK
* NumPy
* Beam Search

## Applications

Potential applications include:

* Accessibility tools for visually impaired users.
* Automatic image annotation.
* Content indexing and retrieval.
* Human-computer interaction.
* Multimedia search systems.
* Intelligent digital assistants.

## Future Improvements

* Attention mechanisms (Bahdanau / Luong).
* Transformer-based decoders.
* Vision Transformers (ViT).
* Larger captioning datasets.
* CLIP-based visual encoders.
* Additional evaluation metrics such as METEOR, CIDEr and SPICE.


