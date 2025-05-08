import pandas as pd
import os
with open("model_list.txt", "r") as f:
    models = f.readlines()
models = [model.replace("\n", "") for model in models]
print(models)

for model in models:    #"deepseek/deepseek-chat-v3-0324", 
    print(model)
    '''
    os.system(f"python prompt_openrouter.py \
              --file del_dataset/del_set_with_QA1.csv \ #del_dataset/del_set_with_QA2.csv \
              --model {model} \
              --content-column QA1 \
              --temperature 0.3")
 
    os.system(f"python prompt_openrouter.py \
              --file final_set/final_set_with_QA_stepwise.csv \
              --model {model} \
              --content-column QA \
              --temperature 0.3 \
              --output-dir final_results/")
    
    os.system(f"python prompt_openrouter.py \
              --file final_set/final_set_with_Qlevel_step.csv \
              --model {model} \
              --content-column QA \
              --temperature 0.3 \
              --output-dir final_results/")
   
    os.system(f"python prompt_openrouter.py \
              --file final_set/final_set_with_plevel_stepv2.csv \
              --model {model} \
              --content-column QA \
              --temperature 0.3 \
              --output-dir final_results/")
    '''
    os.system(f"python prompt_openrouter.py \
              --file final_set/PAR3-final_set_with_QA.csv \
              --model {model} \
              --content-column QA \
              --temperature 0.3 \
              --output-dir final_results/")