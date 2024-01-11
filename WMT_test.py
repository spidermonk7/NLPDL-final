from transformers import TrainingArguments, Trainer
import wandb
from peft import get_peft_model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, TaskType, AdaLoraConfig, IA3Config
import gc
import torch


def run_WMT(tokenized_datasets_train, tokenized_datasets_validation, args):
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = np.argmax(predictions[0], axis=-1)
        decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True).split() for pred in predictions]
        decoded_labels = [tokenizer.decode(label, skip_special_tokens=True).split() for label in labels]

        # Calculate BLEU score
        smooth_fn = SmoothingFunction().method1
        bleu_scores = [sentence_bleu([ref], pred, smoothing_function=smooth_fn) for pred, ref in zip(decoded_preds, decoded_labels)]

        avg_bleu = np.mean(bleu_scores)
        torch.cuda.empty_cache()
        gc.collect()
        return {"bleu": avg_bleu}

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
        run_name = f"WMT(PEFT)_{model_name}_{adpter}"
    else:
        run_name = f"WMT(directly)_{model_name}"

    training_args = TrainingArguments(
        output_dir=f"./results/" + run_name,
        evaluation_strategy="steps",
        save_strategy='epoch',
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay, 
        report_to="wandb"  # Enable logging to W&B
    )

    print("tokenized_datasets_train: ", tokenized_datasets_train)



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_validation,
        tokenizer=tokenizer, 
        compute_metrics=compute_metrics
    )

    if if_report:
        wandb.init(project=project_name, config=training_args)
        wandb.watch(model)

        wandb.run.name = run_name
        trainer.train()

        wandb.finish()
    else:
        trainer.train()


if __name__ == '__main__':
    run_WMT(8)