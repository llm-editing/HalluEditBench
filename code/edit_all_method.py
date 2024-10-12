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
    parser.add_argument('--hparams_dir', default='./hparams', type=str)
    parser.add_argument('--results_dir', default='../results', type=str)
    parser.add_argument('--edit_method', default=None, help='Edit method to use')
    parser.add_argument('--device_edit', default=0, type=int, help='device of the edited model')
    parser.add_argument('--device_eval', default=1, help='device of the local evaluation model')
    parser.add_argument('--dataset_dir', default='../data/questions/hallucination_final', type=str)
    parser.add_argument('--overwrite_result', default=False, action='store_true', help='Overwrite the existing result file')
    parser.add_argument('--model_eval', default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='model id of the local evaluation model')
    parser.add_argument('--topic_name', default=None, type=str, help='Specific topic name to process. If not provided, will process all topics.')
    args = parser.parse_args()
    start_time = time.time()
    topic_name = args.topic_name
    editing_methods = ['LoRA', 'FT-M', 'FT-L', 'ICL', 'ROME', 'MEMIT', 'GRACE']
    if args.edit_method is not None:
        editing_methods = [args.edit_method]

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

        print(f'\nModel: {model_id_format}, Editing {topic_name} with {editing_method}...\n')
        if os.path.exists(f'{args.results_dir}/{model_id_format}/{topic_name}_{editing_method}.json'):
            print(f'Result {topic_name}_{editing_method}.json already exists\n')
            if args.overwrite_result:
                print(f'Overwriting result {topic_name}_{editing_method}.json\n')
            else:
                continue
        df = pd.read_csv(f"{args.dataset_dir}/{model_id_format}/{topic_name}.csv")
        if args.data_size is not None:
            df = df[:args.data_size]
        targets = df['object'].tolist()
        subjects = df['subject'].tolist()
        questions = df['question'].tolist()
        paraphrased_questions = df['paraphrased_question'].tolist()
        locality_questions = {'locality': {'prompt': df['locality_question'].tolist()}}
        df['multiple_choice_full'] = df['question'] + ' ' + df['multiple_choice_with_letters']
        no_questions = {'no': {'prompt': df['no_question'].tolist(), 'ground_truth': ['No' for i in range(len(df))]}}
        yes_questions = {'yes': {'prompt': df['yes_question'].tolist(), 'ground_truth': ['Yes' for i in range(len(df))]}}
        q_and_a_2hop = {'2hop': {'prompt': df['question_2hop'].tolist(), 'ground_truth': df['answer_2hop'].tolist()}}
        q_and_a_3hop = {'3hop': {'prompt': df['question_3hop'].tolist(), 'ground_truth': df['answer_3hop'].tolist()}}
        q_and_a_4hop = {'4hop': {'prompt': df['question_4hop'].tolist(), 'ground_truth': df['answer_4hop'].tolist()}}
        q_and_a_5hop = {'5hop': {'prompt': df['question_5hop'].tolist(), 'ground_truth': df['answer_5hop'].tolist()}}
        q_and_a_6hop = {'6hop': {'prompt': df['question_6hop'].tolist(), 'ground_truth': df['answer_6hop'].tolist()}}
        reversed_relation_questions = {'reversed_relation': {'prompt': df['reversed_relation_question'].tolist(), 'ground_truth': df['subject'].tolist()}}
        multiple_choice_questions = {'multiple_choice': {'prompt': df['multiple_choice_full'].tolist(), 'ground_truth': df['multiple_choice_labels'].tolist()}}

        hparams.device = args.device_edit  # overwrite device in hparams
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            subject=subjects,
            prompts=questions,
            target_new=targets,
            yes_questions=yes_questions,
            no_questions=no_questions,
            locality_inputs=locality_questions,
            rephrase_prompts=paraphrased_questions,
            multiple_choice_questions=multiple_choice_questions,
            reversed_relation_questions=reversed_relation_questions,
            questions_2hop=q_and_a_2hop,
            questions_3hop=q_and_a_3hop,
            questions_4hop=q_and_a_4hop,
            questions_5hop=q_and_a_5hop,
            questions_6hop=q_and_a_6hop,
            summary_metrics=True,
            keep_original_weight=True,
            eval_model_id=args.model_eval,
            device_eval=f'cuda:{args.device_eval}',
            # multi_turn=True,
            # test_generation=True,
        )
        if not os.path.exists(f'{args.results_dir}/{model_id_format}'):
            os.makedirs(f'{args.results_dir}/{model_id_format}')
        # json.dump(metrics, open(f'{args.results_dir}/{model_id_format}/{topic_name}_{editing_method}.json', 'w'), indent=4)
        
        print(f'\nModel: {model_id_format}, Editing {topic_name} with {editing_method} finished')
        del edited_model
        del editor
        gc.collect()
        torch.cuda.empty_cache()

    total_time = (time.time() - start_time) / 60 
    print(f'\nOverall running time for edit_all_method.py: {total_time:.2f} minutes')
# Overall running time for edit_all_method.py: about 240 to 280 minutes