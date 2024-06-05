from transformers import BitsAndBytesConfig

class Config:
    # Model and tokenizer config
    model_name = "meta-llama/Meta-Llama-3-8B"
    saved_model_path = "/content/drive/MyDrive/NLP/QLoRA/trained_model"

    # Dataset config
    dataset_name = "nyu-mll/multi_nli"
    train_subset_ratio = 0.02
    val_subset_ratio = 0.02

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # LoRA config
    lora_rank = 4
    lora_alpha = 16
    lora_dropout = 0.1
    lora_target_modules = ["q_proj", "k_proj", "v_proj"]

    # Training arguments
    output_dir = "/content/drive/MyDrive/NLP/QLoRA/results"
    logging_dir = './logs'
    save_steps = 200
    eval_steps = 50
    logging_steps = 50
    learning_rate = 5e-5
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    weight_decay = 0.01
    save_total_limit = 1
    num_train_epochs = 2
    max_steps = 1000
    report_to = "none"
    load_best_model_at_end = True
    optim = "paged_adamw_8bit"
    fp16 = True
    max_grad_norm = 0.3
    warmup_steps = 2
    group_by_length = True
