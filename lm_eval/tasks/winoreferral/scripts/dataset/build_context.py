import pandas as pd
import json

def build_context(input_csv, output_jsonl): 
    
    context_df = pd.read_csv(input_csv)

    context_df = context_df.reset_index(names='id').melt(id_vars=['id'], var_name='type', value_name='prompt_text')
    context_df.to_json(output_jsonl, orient='records', lines=True)


input_csv = "lm_eval/tasks/winoreferral/data/context.csv"
output_jsonl = "lm_eval/tasks/winoreferral/data/context.jsonl"

build_context(input_csv, output_jsonl)
print(f"Expanded dataset saved to {output_jsonl}")