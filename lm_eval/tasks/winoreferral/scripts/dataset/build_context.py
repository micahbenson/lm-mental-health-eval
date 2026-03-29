import pandas as pd
import json

def build_context(input_csv, output_jsonl): 
    
    context_df = pd.read_csv(input_csv)

    context_df = context_df.reset_index(names='id').melt(id_vars=['id'], var_name='type', value_name='prompt_text')
    context_df.to_json(output_jsonl, orient='records', lines=True)


dep_input_csv = "lm_eval/tasks/winoreferral/data/source/depression_context_final.csv"
dep_output_jsonl = "lm_eval/tasks/winoreferral/data/depression_context.jsonl"

build_context(dep_input_csv, dep_output_jsonl)
print(f"Expanded dataset saved to {dep_output_jsonl}")

anxiety_input_csv = "lm_eval/tasks/winoreferral/data/source/anxiety_context_final.csv"
anxiety_output_jsonl = "lm_eval/tasks/winoreferral/data/anxiety_context.jsonl"

build_context(anxiety_input_csv, anxiety_output_jsonl)
print(f"Expanded dataset saved to {anxiety_output_jsonl}")