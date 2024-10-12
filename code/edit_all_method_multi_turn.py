import os
import gc
import json
import time
import torch
import argparse
import pandas as pd
from hallucination_editor import BaseEditor
from easyeditor import FTHyperParams, IKEHyperParams, ROMEHyperParams, MEMITHyperParams, LoRAHyperParams, GraceHyperParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='llama3-8b')
    parser.add_argument('--data_size', default=None, type=int)
    parser.add_argument('--topic_name', default=None, type=str)
    parser.add_argument('--hparams_dir', default='./hparams', type=str)
    parser.add_argument('--results_dir', default='../results', type=str)
    parser.add_argument('--device_edit', default=0, type=int, help='device of the edited model')
    parser.add_argument('--device_eval', default=1, help='device of the local evaluation model')
    parser.add_argument('--dataset_dir', default='../data/questions/hallucination_final', type=str)
    parser.add_argument('--multi_turn', default='yes', choices=['yes', 'sure'], help='Type of multi-turn evaluation')
    parser.add_argument('--overwrite_result', default=False, action='store_true', help='Overwrite the existing result file')
    parser.add_argument('--model_eval', default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='model id of the local evaluation model')
    parser.add_argument('--editing_method', default=None, type=str, help='Specific editing method to use. If not provided, will process all methods.')
    args = parser.parse_args()
    start_time = time.time()

    if args.editing_method:
        editing_methods = [args.editing_method]
    else:
        editing_methods = ['LoRA', 'MEMIT', 'FT-M', 'FT-L', 'ICL', 'ROME', 'GRACE']

    for editing_method in editing_methods:
        if editing_method in ['FT-M', 'FT-L']:
            editing_hparams = FTHyperParams
        elif editing_method == 'ICL':
            editing_hparams = IKEHyperParams
        elif editing_method == 'ROME':
            editing_hparams = ROMEHyperParams
        elif editing_method == 'MEMIT':
            editing_hparams = MEMITHyperParams
        elif editing_method == 'LoRA':
            editing_hparams = LoRAHyperParams
        elif editing_method == 'GRACE':
            editing_hparams = GraceHyperParams
        else:
            raise NotImplementedError

        hparams = editing_hparams.from_hparams(f'{args.hparams_dir}/{editing_method}/{args.model_name}')
        model_id_format = hparams.model_name.split('/')[-1].replace('-', '_').lower()

        topic_name = args.topic_name
        results_dir = f'{args.results_dir}/{model_id_format}_multi_turn'
        results_file_name = f'{topic_name}_{editing_method}_{args.multi_turn}.json'
        print(f'Model: {model_id_format}, Editing {topic_name} with {editing_method}...\n')
        if os.path.exists(f'{results_dir}/{results_file_name}'):
            print(f'Result {results_file_name} already exists\n')
            if args.overwrite_result:
                print(f'Overwriting result {results_file_name}\n')
            else:
                continue
        df = pd.read_csv(f"{args.dataset_dir}/{model_id_format}/{topic_name}.csv")
        if args.data_size is not None:
            df = df[:args.data_size]
        targets = df['object'].tolist()
        subjects = df['subject'].tolist()
        questions = df['question'].tolist()
        no_questions = {'no': {'prompt': df['no_question'].tolist(), 'ground_truth': ['No' for i in range(len(df))]}}
        yes_questions = {'yes': {'prompt': df['yes_question'].tolist(), 'ground_truth': ['Yes' for i in range(len(df))]}}

        hparams.device = args.device_edit  # overwrite device in hparams
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            subject=subjects,
            prompts=questions,
            target_new=targets,
            yes_questions=yes_questions,
            no_questions=no_questions,
            summary_metrics=True,
            keep_original_weight=True,
            eval_model_id=args.model_eval,
            device_eval=f'cuda:{args.device_eval}',
            multi_turn=args.multi_turn,
        )
        if not os.path.exists(f'{results_dir}'):
            os.makedirs(f'{results_dir}')
        json.dump(metrics, open(f'{results_dir}/{results_file_name}', 'w'), indent=4)
        
        del edited_model
        del editor
        gc.collect()
        torch.cuda.empty_cache()

    total_time = (time.time() - start_time) / 60
    print(f'\nOverall running time (Model: {model_id_format}, Editing {topic_name} with 7 editing_method): {total_time:.2f} minutes')
# Overall running time (Model: llama_2_7b_chat_hf, Editing art_sculpture with 7 editing_method): 384.13 minutes