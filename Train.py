from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi23",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    warmup_steps=1000,
    max_steps=5000,
    gradient_checkpointing=False,
    eval_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_strategy='steps',
    save_steps=2500,
    eval_steps=2500,
    remove_unused_columns=False,
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)
