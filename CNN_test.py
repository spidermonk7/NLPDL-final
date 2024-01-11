# Import necessary libraries
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
# Weights & Biases Setup
import wandb
from rouge_score import rouge_scorer
from peft import LoraConfig, TaskType, AdaLoraConfig, LoHaConfig, LoKrConfig, IA3Config
from peft import get_peft_model
import torch
from rouge_score import rouge_scorer
import numpy as np
import gc
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def run_CNN(tokenized_datasets_train, tokenized_datasets_validation, args):
    peftt = args.peftt
    adpter = args.adpter
    model_name = args.model_name
    r = args.r
    if_report = args.report_to_wandb
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    project_name = args.project_name

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = np.argmax(predictions[0], axis=-1)
        torch.no_grad()
        # print(f'prediction shape: {predictions.shape}, label shape {labels.shape}')
        # Decode the predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(pred.strip().split('.')) for pred in decoded_preds]
        decoded_labels = ["\n".join(label.strip().split('.')) for label in decoded_labels]

        # Initialize Rouge scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        # Compute ROUGE scores
        rouge_scores = [scorer.score(label, pred) for label, pred in zip(decoded_labels, decoded_preds)]

        # Aggregate scores
        rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        rouge2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)

        torch.cuda.empty_cache()
        gc.collect()
        return {"rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL}

    print("dataset loaded")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print("tokenizer and model loaded: ", model_name)



    if adpter == 'lora':
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            lora_alpha=32,
            lora_dropout=0.1,
            r=r
        )
    elif adpter == 'adalora':
        peft_config = AdaLoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            lora_alpha=32,
            lora_dropout=0.1,
            r=r
        )
    elif adpter == 'IA3':
        peft_config = IA3Config(
        peft_type="IA3",
        task_type="SEQ_2_SEQ_LM",
        target_modules=["k", "v", "w0"],
        feedforward_modules=["w0"],
        )
    
    else:
        raise ValueError("adpter not found")


    if peftt:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    else: 
        model = model

    if peftt:
        run_name = f"CNN(PEFT)_{model_name}_{adpter}"
    else:
        run_name = f"CNN(directly)_{model_name}"

    print("dataset preprocessed")
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./results/" + run_name,
        evaluation_strategy="steps",
        save_strategy='epoch', 
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        # report_to="wandb",  # Enable logging to W&B
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_validation,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    print("trainer initialized, starting training")

    if if_report:
        # Initialize Weights & Biases
        wandb.init(project=project_name, config=training_args)
        wandb.watch(model)

        wandb.run.name = run_name

        # # Start Training
        trainer.train()

        wandb.finish()
    else:
        trainer.train()


if __name__ == '__main__':
    
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
        
    Debugging = False
    if not Debugging:     
        # Load the CNN/DailyMail dataset
        CNNdataset_train = load_dataset("cnn_dailymail", "3.0.0", split="train[:10%]")
        CNNdataset_validation = load_dataset("cnn_dailymail", "3.0.0", split="validation[:5%]")
    else:
        CNNdataset_train = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
        CNNdataset_validation = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1%]")


    # Function to preprocess the data
    def CNNpreprocess_function(examples):
        inputs = ["summarize: " + doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        # make it a dictionary of train and validation

        return model_inputs



    CNNtokenized_datasets_train = CNNdataset_train.map(CNNpreprocess_function, batched=True)
    CNNtokenized_datasets_validation = CNNdataset_validation.map(CNNpreprocess_function, batched=True)
    r = 8
    run_CNN(CNNtokenized_datasets_train, CNNtokenized_datasets_validation, r=r, adpter='IA3')