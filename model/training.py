from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

def train_model(model, tokenizer, tokenized_datasets, config):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        logging_dir=config.logging_dir,
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        weight_decay=config.weight_decay,
        save_total_limit=config.save_total_limit,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        report_to=config.report_to,
        load_best_model_at_end=config.load_best_model_at_end,
        optim=config.optim,
        fp16=config.fp16,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=config.warmup_steps,
        group_by_length=config.group_by_length,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation_matched'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the model with LoRA weights
    model.save_pretrained(config.saved_model_path)
    tokenizer.save_pretrained(config.saved_model_path)

    # Manually merge LoRA weights with the initial weights of the model
    def merge_lora_weights(model):
        for name, param in model.named_parameters():
            if "lora" in name:
                original_name = name.replace("lora", "weight")
                if hasattr(model, original_name):
                    original_param = getattr(model, original_name)
                    original_param.data += param.data
                    param.data = original_param.data

    merge_lora_weights(model)

    # Save the merged model
    model.save_pretrained(config.saved_model_path)
