# Natural Language Processing: NLI Model Fine-Tuning and Optimization

## Project Overview

Natural Language Inference (NLI) is a core task in NLP, where the model determines the logical relationship between two given sentences: entailment, contradiction, or neutral. This project examines different techniques to optimize the fine-tuning process of large language models, focusing on achieving high accuracy with minimal computational costs.

This repository contains the code for training a LLaMA3 model using QLoRA for the task of Natural Language Inference (NLI) on the MultiNLI dataset.

## Datasets

The project uses popular NLI datasets:
- **MultiNLI**: Includes diverse genres with approximately 433k sentence pairs.
- **SNLI**: Similar to MultiNLI but less diverse in genre.

Each pair is labeled with one of three categories:
- **Entailment**: One sentence logically follows from the other.
- **Contradiction**: The sentences are logically contradictory.
- **Neutral**: No clear logical relationship.

## Models and Techniques

We experimented with multiple fine-tuning and optimization strategies on the following models:

### 1. Full Fine-Tuning
In this method, the entire set of model parameters is adjusted during training. This approach is computationally intensive, especially for large models, and requires significant memory resources.

### 2. LoRA (Low-Rank Adaptation)
LoRA reduces the number of trainable parameters by freezing most layers of the model and training only a few low-rank matrices. This approach significantly decreases the computational load while retaining high performance.

### 3. P-tuning
P-tuning employs "soft prompts," which are optimized embeddings that guide the model’s behavior without explicit textual prompts. This method allows for flexible, interpretable prompting.

### 4. QLoRA
QLoRA is an enhanced LoRA-based approach that quantizes weights to further reduce memory requirements and computation costs while preserving performance. It leverages efficient training with quantized 4-bit or 8-bit precision.


## Implementation

### Prerequisites
To replicate this project, you will need:
- **Python 3.8+**
- **Transformers** library by Hugging Face
- **Torch**
- **scikit-learn**
- **GPU** with CUDA support for faster model training (optional but recommended)

## Project Structure
- **`NLP_CA4_Q1.ipynb`**: Notebook for the initial experiments with RoBERTa on MultiNLI and SNLI datasets, using various fine-tuning strategies.
- **`NLP_CA4_Q2.ipynb`**: Notebook focusing on Llama3 8B model optimization using LoRA, P-tuning, and QLoRA.
- **`report.pdf`**: Detailed documentation of the experiments, analysis, and results.


### Running Experiments
1. Open each notebook in Jupyter or Colab.
2. Follow the cells sequentially to reproduce the experiments and analyses.
3. Adjust hyperparameters or modify configurations to experiment with different settings.


### Key Hyperparameters
- **Learning Rate**: Optimized based on each model's behavior, typically starting at `1e-5` to `1e-4`.
- **Batch Size**: Adjusted based on GPU memory constraints; optimal sizes vary across techniques.
- **Number of Virtual Tokens** (for P-tuning): Set to 30 for this project.
- **Quantization Bits** (for QLoRA): 4-bit or 8-bit quantization was tested to balance performance and efficiency.

## Results and Findings

| Model       | Method           | Trainable Params | Training Time | Accuracy |
|-------------|------------------|------------------|---------------|----------|
| RoBERTa     | Full Fine-Tuning | 360M            | 1h 39m       | 32.5%    |
| RoBERTa     | LoRA             | 1.6M            | 1h 16m       | 83.5%    |
| RoBERTa     | P-tuning         | 4.2M            | 1h 09m       | 36%      |
| Llama3 8B   | Zero-shot        | -               | -            | 40.5%    |
| Llama3 8B   | One-shot         | -               | -            | 26.7%    |
| Llama3 8B   | QLoRA            | 8M              | 1h 50m       | 81.8%    |

- **LoRA** provided the best balance between training speed and accuracy for the RoBERTa model.
- **QLoRA** showed strong results for Llama3 8B, with substantial memory savings.
- **Zero-shot and One-shot** prompting yielded lower accuracy, indicating the need for model fine-tuning on NLI tasks.

## Conclusion

This project demonstrates that low-rank adaptation techniques (e.g., LoRA) and quantization-based optimizations (e.g., QLoRA) can significantly enhance model performance on NLI tasks while minimizing memory and computational costs. Future work may include testing with larger, more diverse datasets and exploring other advanced prompting strategies.

## References
- Hugging Face Transformers: [Transformers Library](https://huggingface.co/transformers/)
- Original papers on LoRA, QLoRA, and P-tuning for additional insights.

---

## License

This project is licensed under the MIT License.

