# NLPDL-final: Can we teach a model twice?
This is a repository for my NLPDL course final project.



<you might add a picture here>

## Quick start
You should only consider 2 file: ***main.py*** and ***retune_mian.py***.

If you want to tune a model with lora for the first time \
``python main.py --model_name=t5-small --dataset=CNN --peftt=True --adpter=lora ``


And for the retuning process, you can use:\
``python retune_main.py --model_name=t5-small --dataset=CNN --peftt=True --adpter=lora ``


Here we offer all the parser configurations and their defaults, just modify them based on your requirements.


