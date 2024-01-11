from BBC_test import run_BBC
from CNN_test import run_CNN
from WMT_test import run_WMT
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import argparse



def load_parser():
    parser = argparse.ArgumentParser(description='PEFTT')
    parser.add_argument('--model_name', type=str, default='t5-small', help='model name')
    parser.add_argument('--peftt', type=bool, default=True, help='whether to use PEFTT')
    parser.add_argument('--adpter', type=str, default='lora', help='adapter type')
    parser.add_argument('--Debugging', type=bool, default=False, help='whether to use Debugging')
    parser.add_argument('--data_name', type=str, default='WMT', help='data name')
    parser.add_argument('--r', type=int, default=32, help='hidden size of lora')
    parser.add_argument('--report_to_wandb', type=bool, default=True, help='whether to report to wandb')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--project_name', type=str, default='NLPDL-FINAL', help='project name')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    return parser





if __name__ == '__main__':
    model_name_list = ['t5-small', 't5-base', 't5-large', 'facebook/bart-base']
    parser = load_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model_name = args.model_name
    r = args.r
    adpter = args.adpter
    Debugging = args.Debugging
    data_name = args.data_name

    # validation check
    assert model_name in model_name_list
    assert data_name in ['WMT', 'BBC', 'CNN']
    assert adpter in ['lora', 'adalora', 'IA3']


    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # here we carry out training procedure based on the data_name
    if data_name == 'WMT':
        if not Debugging:
            # # only use 50% of the trainset
            WMTtrain_dataset = load_dataset("wmt16", "de-en", split="train[:3%]")
            WMTvalidation_dataset = load_dataset("wmt16", "de-en", split="validation[:2%]")
        else:
            WMTtrain_dataset = load_dataset("wmt16", "de-en", split="train[:1%]")
            WMTvalidation_dataset = load_dataset("wmt16", "de-en", split="validation[:1%]")
   
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
        print(f"running model {model_name} on dataset WMT" )
        run_WMT(tokenized_datasets_train=WMTtokenized_datasets_train, tokenized_datasets_validation=WMTtokenized_datasets_validation, args=args)


    elif data_name == 'BBC':
        if not Debugging:

            # Load the XSum dataset
            BBCdataset_train = load_dataset("xsum", split="train[:10%]")
            BBCdataset_validation = load_dataset("xsum", split="validation[:5%]")

        else:
            BBCdataset_train = load_dataset("xsum", split="train[:1%]")
            BBCdataset_validation = load_dataset("xsum", split="validation[:1%]")


        # Function to preprocess the data
        def BBCpreprocess_function(examples):
            # print(examples)

            inputs = ["summarize: " + doc for doc in examples["document"]]
            model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")


            model_inputs["labels"] = labels["input_ids"]
            # make it a dictionary of train and validation

            return model_inputs


        BBCtokenized_datasets_train = BBCdataset_train.map(BBCpreprocess_function, batched=True)
        BBCtokenized_datasets_validation = BBCdataset_validation.map(BBCpreprocess_function, batched=True)
        print("dataset preprocessed: BBC")

        print(f"running model {model_name} on dataset BBC")
        run_BBC(tokenized_datasets_train=BBCtokenized_datasets_train, tokenized_datasets_validation=BBCtokenized_datasets_validation, args=args)


    elif data_name == 'CNN':
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
        print("dataset preprocessed: CNN")
        print(f"running model {model_name} on dataset CNN" )
        run_CNN(tokenized_datasets_train=CNNtokenized_datasets_train, tokenized_datasets_validation=CNNtokenized_datasets_validation, args=args)



    
   
   
    
