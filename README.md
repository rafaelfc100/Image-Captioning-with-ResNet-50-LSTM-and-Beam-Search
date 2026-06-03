Image Captioning with ResNet-50, LSTM and Beam Search

This project implements an Image Captioning system capable of automatically generating natural language descriptions from images. The architecture follows the Encoder-Decoder paradigm, combining a pretrained ResNet-50 for visual feature extraction and a two-layer LSTM for sentence generation.

Unlike the original Show and Tell architecture, this implementation injects the visual feature vector into the decoder at every generation step, allowing the model to maintain visual context throughout the captioning process. During inference, Beam Search is used to improve caption quality compared to greedy decoding.

Project Objectives
Generate natural language descriptions from images.
Combine Computer Vision and Natural Language Processing techniques.
Evaluate caption quality using BLEU metrics and perplexity.
Compare greedy decoding against Beam Search.
Deploy the model through a web application for real-time inference.
Dataset

The model was trained using the Flickr30k dataset, which contains images paired with human-written captions.

Split	Images	Captions
Training	25,426	127,129
Validation	3,178	15,890
Test	3,179	15,895

A vocabulary of 11,420 tokens was built after filtering rare words and including special tokens:

<PAD>
<SOS>
<EOS>
<UNK>
Model Architecture
Encoder
ResNet-50 pretrained on ImageNet
Output feature vector: 2048 dimensions
Batch Normalization
Dropout Regularization
Projection into embedding space
Decoder
Two-layer LSTM
Hidden size: 512
Embedding size: 256
Visual feature injection at every decoding step
Teacher Forcing during training
Decoding Strategies
Greedy Search (Beam = 1)
Beam Search (Beam = 3)
Beam Search (Beam = 5)
Beam Search (Beam = 7)
Training Configuration
Hyperparameter	Value
Optimizer	Adam
Learning Rate	4e-4
Batch Size	128
Weight Decay	1e-4
Dropout	0.5
Gradient Clipping	5.0
Epochs	20 / 30
Results
Validation Performance
Metric	Value
Validation Loss	2.9466
Perplexity	19.03
BLEU Scores
Metric	Score
BLEU-1	0.6386
BLEU-2	0.4304
BLEU-3	0.2988
BLEU-4	0.2035
Inference Speed
Metric	Value
Images per Second	7.98
Key Contributions
Visual Injection

Instead of providing image information only at the beginning of the decoder, the visual feature vector is concatenated with every word embedding during sentence generation. This improves coherence between generated captions and image content.

Beam Search

Beam Search maintains multiple candidate sentences simultaneously and selects the best sequence according to normalized probability scores. This produces more complete and fluent descriptions than greedy decoding.

Web Application

A Flask-based web interface was developed to make the model accessible through a browser.

Features include:

Image upload via file selection or drag-and-drop
Real-time caption generation
Multiple decoding modes
Inference time measurement
CPU/GPU execution information
External image testing
Example Output

Input Image:

People working on a construction site

Generated Caption:

"a group of construction workers working on a train"

Future Improvements
Attention Mechanisms (Bahdanau / Luong)
Transformer-based Decoders
Vision Transformers (ViT)
Larger captioning datasets
CLIP-based visual encoders
BLEU, METEOR, CIDEr and SPICE evaluation
Technologies
Python
PyTorch
Torchvision
ResNet-50
LSTM
Flask
NumPy
NLTK
Beam Search
