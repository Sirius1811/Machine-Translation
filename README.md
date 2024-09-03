# Transformer-Based Machine Translation Model
This repository contains an implementation of a transformer-based machine translation model, inspired by the "Attention is All You Need" paper. The model is designed to translate text from English to Italian using the Opus Books dataset.

## Project Overview
This project demonstrates the implementation of a transformer model for sequence-to-sequence tasks, specifically for machine translation. The model is built using PyTorch and incorporates key components such as multi-head self-attention, positional encoding, and feed-forward neural networks.

### Key Features:
**Custom Bilingual Dataset Class:** Handles tokenization, sequence padding, and prepares data for training.
**Multi-Head Attention:** Implements the attention mechanism that allows the model to focus on different parts of the input sequence.
**Positional Encoding:** Adds positional information to input embeddings to capture the order of words in the sequences.
**Layer Normalization & Residual Connections:** Stabilizes training and allows for deeper networks.

## Repository Structure
**config.py:** Contains configuration settings for the model, including hyperparameters like batch size, learning rate, and sequence length. It also includes utility functions to manage model weights and paths.

**dataset.py:** Defines the BilingualDataset class, responsible for preprocessing the dataset, tokenizing the source and target languages, and generating input sequences for the encoder and decoder.

**model.py:** Implements the core components of the transformer model, including multi-head attention, positional encoding, encoder, decoder, and the final projection layer. The model is built to translate from one language to another.

## Installation
**Clone the repository:**
git clone https://github.com/Sirius1811/Machine-Translation.git
cd transformer-translation

**Install dependencies:**
pip install -r requirements.txt

**Download the dataset:** This model uses the Opus Books (Hugging Face dataset) for training. You can download it using the link below

## Usage
**Training the Model**
To train the model, run the following command:
python train.py
This script will load the dataset, prepare the data, and start the training process. You can adjust the hyperparameters in the config.py file.

**Evaluating the Model**
To evaluate the model on a test dataset:
python evaluate.py --weights_file path/to/weights

**Configuration**
The config.py file contains all the configuration settings for the model, including:

batch_size: Batch size used during training.

num_epochs: Number of epochs for training.

lr: Learning rate for the optimizer.

seq_len: Sequence length for input and output sentences.

d_model: Dimensionality of the model.

datasource: Name of the data source used for training.

lang_src: Source language code (e.g., 'en').

lang_tgt: Target language code (e.g., 'it').

model_folder: Folder where model weights are saved.

model_basename: Base name for saved model files.

preload: Option to preload the latest weights.

## Model Architecture
The transformer model consists of the following components:

**Input Embeddings:** Converts tokens into dense vectors.

**Positional Encoding:** Adds positional information to the embeddings.

**Multi-Head Attention:** Allows the model to focus on different parts of the input.

**Feed-Forward Neural Networks:** Provides non-linearity and depth to the model.

**Encoder and Decoder Blocks:** Stack of layers that process the input sequence and generate the output sequence.

**Projection Layer:** Maps the output of the decoder to the target vocabulary.

## Future Work
Experiment with Different Datasets: Apply the model to different language pairs and datasets.
Hyperparameter Tuning: Explore different configurations of the transformer model.
Inference Pipeline: Create an easy-to-use inference pipeline for real-time translation.

## Acknowledgements
The implementation is based on the "Attention is All You Need" paper by Vaswani et al.
Dataset provided by the [Opus Books project](https://huggingface.co/datasets/Helsinki-NLP/opus_books).
