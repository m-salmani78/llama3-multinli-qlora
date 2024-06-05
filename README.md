# LLaMA3 with QLoRA for Natural Language Inference

This repository contains the code for training a LLaMA3 model using QLoRA for the task of Natural Language Inference (NLI) on the MultiNLI dataset. The model predicts the relationship between a pair of sentences: entailment, contradiction, or neutral.

## Project Overview

This project demonstrates how to:
1. Load and preprocess the MultiNLI dataset.
2. Configure and train a LLaMA3 model with LoRA for efficient fine-tuning.
3. Merge the LoRA weights with the original model weights after training.
4. Evaluate the model's performance on the validation dataset.

## Requirements

To run this project, you will need the following packages:
- `transformers`
- `datasets`
- `torch`
- `peft`
- `tqdm`
- `matplotlib`
- `numpy`
- `warnings`

You can install the necessary packages using `pip`:
```bash
pip install transformers datasets torch peft tqdm matplotlib numpy
```

## Project Structure

- `main.py`: The main script to train and evaluate the model.
- `README.md`: Project description and instructions.

## Usage

### Training the Model

The `main.py` script includes the full pipeline for training the LLaMA3 model with LoRA on the MultiNLI dataset.

1. **Preprocess the Data**:
   The preprocessing function formats the input pairs and tokenizes them.

2. **Configure the Model**:
   Load the LLaMA3 model and apply the LoRA configuration.

3. **Train the Model**:
   Use the `Trainer` class from the `transformers` library to train the model.

4. **Merge LoRA Weights**:
   After training, merge the LoRA weights with the original model weights.

5. **Evaluate the Model**:
   Evaluate the model's performance on the validation set.

### Running the Script

To train and evaluate the model, run:
```bash
python main.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License.

