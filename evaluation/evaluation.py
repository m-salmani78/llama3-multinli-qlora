import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_model(model, tokenizer, validation_dataset, id2label, config):
    model.eval()
    total, correct = 0, 0
    for example in validation_dataset:
        inputs = tokenizer(f"[INST] {example['premise']} [SEP] {example['hypothesis']} [/INST]", return_tensors='pt', max_length=300, padding='max_length', truncation=True)
        labels = example['label']
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=350)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).split('NLI Label: ')[-1].strip()
        if prediction == id2label[labels]:
            correct += 1
        total += 1
    accuracy = correct / total
    return accuracy
