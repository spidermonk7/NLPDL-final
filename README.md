# NLPDL-final: Can we teach a model twice?
This is a repository for my NLPDL course final project.



<you might add a picture here>

## Quick start
You should only consider 2 file: ***main.py*** and ***retune_mian.py***.


### Tune your model for the first time
If you want to tune a model with lora for the first time \
``python main.py --model_name=t5-small --dataset=CNN --peftt=True --adpter=lora ``




Here we offer all the parser configurations and their defaults, just modify them based on your requirements.

| config | default | --help |
|:------|:-------:|------:|
| --model_name | t5-small | model name |
| --peftt | True | Implement PEFT or not |
| --adpter | lora | Which peft method |
| --Debugging | False | If debugging or running |
| --data_name | WMT | The dataset |
| --r | 32 | hidden size of lora |
| --report_to_wandb | True | Report to wandb or not |
| --epochs | 3 | num of epochs |
| --batch_size | 16 | batch size |
| --lr | 1e-5 | leaerning rate |
| --weight_decay | 0.01 | weight decay |
| --project_name | NLPDL-FINAL | Project name on W&B |

### Storage Path
We didn't offer a config for storage_path in the parsing arguments. The default is:


**Tune directly:** [  ./results/{data_name}(directly)\_{model_name}  ]


***Tune with peft:*** [  ./results/{data_name}(PEFT)\_{model_name}\_{adpter}  ]


### valid models, datasets and adapters
Here are the valid models that u can use. A undefined model/dataset/adpter would raise an error.

**model\_name:** t5-small, t5-base, t5-large, facebook/bart-base

**dataset\_name:** CNN, BBC, WMT 

**adpter:** lora, adalora, IA3

**Attention: Please don't use IA3 on Bart.**




### Tune your model for the second time

And for the retuning process, you can use:\
``python retune_main.py --model_name=t5-small --dataset=CNN --peftt=True --adpter=lora ``

Also, we offer all the parser configurations and their defaults for retuning, just modify them based on your requirements.

| config | default | --help |
|:------|:-------:|------:|
| --model_name | t5-small | model name |
| --peftt | True | Implement PEFT or not |
| --adpter | lora | Which peft method |
| --Debugging | False | If debugging or running |
| --data_name | WMT | The dataset |
| --r | 32 | hidden size of lora |
| --report_to_wandb | True | Report to wandb or not |
| --epochs | 3 | num of epochs |
| --batch_size | 16 | batch size |
| --lr | 1e-5 | leaerning rate |
| --weight_decay | 0.01 | weight decay |
| --project_name | NLPDL-FINAL | Project name on W&B |

