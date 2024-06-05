from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def create_model(config):
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(config.model_name, quantization_config=config.bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True

    # Define the LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    return model, tokenizer
