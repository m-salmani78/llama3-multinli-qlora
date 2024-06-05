from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess_data(config):
    # Load the MultiNLI dataset
    dataset = load_dataset(config.dataset_name)
    class_names = dataset["train"].features["label"].names
    id2label = {i: label for i, label in enumerate(class_names)}
    label2id = {label: i for i, label in enumerate(class_names)}

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True

    # Preprocessing function
    def preprocess_function(examples):
        inputs = [f"[INST] {premise} [SEP] {hypothesis} [/INST] NLI Label: {id2label[label_id]}" for premise, hypothesis, label_id in zip(examples['premise'], examples['hypothesis'], examples['label'])]
        model_inputs = tokenizer(inputs, max_length=300, padding='max_length', truncation=True)
        return model_inputs

    # Preprocess the datasets
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)
    
    return tokenized_datasets, id2label
