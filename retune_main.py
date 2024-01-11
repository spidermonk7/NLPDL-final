from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import wandb
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gc
import argparse

def load_parser():
    parser = argparse.ArgumentParser(description='PEFTT')
    parser.add_argument('--model_name', type=str, default='t5-small', help='model name')
    parser.add_argument('--peftt', type=bool, default=True, help='whether to use PEFTT')
    parser.add_argument('--adpter', type=str, default='lora', help='adapter type')
    parser.add_argument('--Debugging', type=bool, default=False, help='whether to use Debugging')
    parser.add_argument('--data_name', type=str, default='WMT', help='data name, in [WMTT, BBC]')
    parser.add_argument('--r', type=int, default=32, help='hidden size of lora')
    parser.add_argument('--report_to_wandb', type=bool, default=True, help='whether to report to wandb')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--project_name', type=str, default='NLPDL-FINAL', help='project name')
    parser.add_argument('--checkpoint', type=int, default=5385, help='checkpoint')
    parser.add_argument('--seed', type=int, default=2023, help='seed')
    return parser



def run_retune(args):
    Debugging = args.Debugging  
    model_name = args.model_name
    data_name = args.data_name
    peft = args.peftt
    checkpoint = args.checkpoint
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    project_name = args.project_name
    adpter = args.adpter
    checkpoint = args.checkpoint

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if data_name == 'WMT':
        if not Debugging:
            # # only use 50% of the trainset
            WMTtrain_dataset = load_dataset("wmt16", "de-en", split="train[:3%]")
            WMTvalidation_dataset = load_dataset("wmt16", "de-en", split="validation[:2%]")
        else:
            WMTtrain_dataset = load_dataset("wmt16", "de-en", split="train[:1%]")
            WMTvalidation_dataset = load_dataset("wmt16", "de-en", split="validation[:1%]")


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
        
        def wmtpreprocess_translation(examples):
            inputs = []
            targets = []
            for ittem in examples['translation']:
                inputs.append("translate English to German: " + ittem['en'])
                targets.append(ittem['de'])

            model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        WMTtokenized_datasets_train = WMTtrain_dataset.map(wmtpreprocess_translation, batched=True)
        WMTtokenized_datasets_validation = WMTvalidation_dataset.map(wmtpreprocess_translation, batched=True)
        print("dataset preprocessed: WMT")
        train_dataset = WMTtokenized_datasets_train
        validation_dataset = WMTtokenized_datasets_validation

    elif data_name == 'BBC':
        if not Debugging:
            BBCdataset_train = load_dataset("xsum", split="train[:10%]")
            BBCdataset_validation = load_dataset("xsum", split="validation[:5%]")
        else:
            BBCdataset_train = load_dataset("xsum", split="train[:1%]")
            BBCdataset_validation = load_dataset("xsum", split="validation[:1%]")
        def compute_metrics(eval_preds):
            predictions, labels = eval_preds
            predictions = np.argmax(predictions[0], axis=-1)
           
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

        def BBCpreprocess_function(examples):
            # print(examples)
            inputs = []
            targets = []
            for ittem in examples['document']:
                inputs.append("summarize: " + ittem)
            for ittem in examples['summary']:
                targets.append(ittem)

            model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        BBCtokenized_datasets_train = BBCdataset_train.map(BBCpreprocess_function, batched=True)
        BBCtokenized_datasets_validation = BBCdataset_validation.map(BBCpreprocess_function, batched=True)
        print("dataset preprocessed: BBC")
        train_dataset = BBCtokenized_datasets_train
        validation_dataset = BBCtokenized_datasets_validation

    if peft == False:
        # load trained model from checkpoint
        path = f"results/CNN(directly)_{model_name}/checkpoint-{checkpoint}"

    else:
        path = f"results/CNN(PEFT)_{model_name}_{adpter}/checkpoint-{checkpoint}"


    print(f"loading checkpoint {checkpoint}")
    # load parameters from checkpoint
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    if peft == True:
        if args.adpter == 'lora':
            for name, param in model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True

    print("model loaded")

    Train_Config = TrainingArguments(
        output_dir=f"./CNN_retune/{data_name}_{model_name}_{peft}",
        evaluation_strategy="steps",
        save_strategy='no',
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay
    )

    trainer = Trainer(
        model=model,
        args=Train_Config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    if args.report_to_wandb:
        wandb.init(project=project_name, config=Train_Config)
        wandb.watch(model)

        wandb.run.name = f"PEFT_{data_name}_{model_name}_retune_checkpoint{checkpoint}"
        trainer.train()
        wandb.finish()
    else:
        trainer.train()


if __name__ == '__main__':
    parser = load_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)


    checkpoint_list = [1795, 3590, 5385]

    run_retune(args)
