from config import Config
from data.preprocessing import load_and_preprocess_data
from model.model_setup import create_model
from model.training import train_model
from evaluation.evaluation import evaluate_model

def main():
    config = Config()
    
    # Load and preprocess the data
    tokenized_datasets, id2label = load_and_preprocess_data(config)

    # Create the model
    model, tokenizer = create_model(config)

    # Train the model
    train_model(model, tokenizer, tokenized_datasets, config)

    # Evaluate the model
    accuracy = evaluate_model(model, tokenizer, tokenized_datasets['validation_matched'], id2label, config)
    print(f"Validation Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
