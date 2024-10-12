import json
import math
import torch
import random
import typing
import logging
import numpy as np
from tqdm import tqdm
from time import time
from typing import Optional, Union, List, Dict
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from easyeditor.util import nethook
from easyeditor.util.globals import *
from easyeditor.util.alg_dict import *
from easyeditor.models.melo.melo import LORA
from easyeditor.util.hparams import HyperParams
from easyeditor.editors.batch_editor import BatchEditor
from easyeditor.evaluate.evaluate_utils import test_generation_quality
from easyeditor.evaluate import compute_icl_edit_quality, compute_sent_metric

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO) # filemode="w",
LOG = logging.getLogger(__name__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
system_msg_eval = "Given two texts, labeled as Text 1 and Text 2, output '1' if they match each other semantically; otherwise, output '0'. Do not repeat the question or provide any explanation."   
system_msg_qa = "Always respond to the input question concisely with a short phrase or a single-word answer. Do not repeat the question or provide any explanation."


def make_logs():
    # f_h, s_h = get_handler('logs', log_name='editing.log')
    # LOG.addHandler(f_h)
    s_h = logging.StreamHandler()
    LOG.addHandler(s_h)


def get_all_acc_keys(dict_list):
    all_keys = set()

    def recursive_keys(d):
        for k, v in d.items():
            if k.endswith('acc'):
                all_keys.add(k)
            if isinstance(v, dict):
                recursive_keys(v)
                
    for dictionary in dict_list:
        recursive_keys(dictionary)

    return all_keys


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(42)


def get_response(hparams, model, tok, messages, max_new_tokens=1, eval_flag=False, device_eval='cuda:0'): 
    device = device_eval if eval_flag else hparams.device
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(device)
    output_ids = model.generate(**msg_tokenized, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(output_ids[0][msg_tokenized['input_ids'].shape[-1]:], skip_special_tokens=True).replace('\n', ' ').strip().rstrip('.')


def evaluate_response(hparams, model_eval, tok_eval, prompt_qa, output_qa, label, device_eval):
    if output_qa.lower() in label.lower() or label.lower() in output_qa.lower():  # Exact and partial match
        response_eval = 1
    else:  # Semantic match
        user_msg_eval = f"""Text 1: {label} \nText 2: {output_qa}"""
        messages_eval = [{"role": "system", "content": system_msg_eval}, {"role": "user", "content": user_msg_eval}]
        response_eval = get_response(hparams, model_eval, tok_eval, messages_eval, eval_flag=True, device_eval=device_eval)

    print(f"===== Question: {prompt_qa} | Prediction: {output_qa} | Label: {label} | Evaluation: {response_eval} =====")  #  (1 denotes correct)
    if str(response_eval) not in ['0', '1']:
        response_eval = 0
    return int(response_eval), output_qa


def test_prediction_acc(hparams, model_qa, tok_qa, model_eval, tok_eval, device_eval, prompt_qa, label, pre_or_post, vanilla_generation=False, system_msg=system_msg_qa):
    if vanilla_generation and pre_or_post=='post':
        # target_new_tokens = tok_qa.encode(label, add_special_tokens=False) 
        # prompt_tok = tok_qa.apply_chat_template([{"role": "user", "content": prompt_qa}], add_generation_prompt=True, return_tensors='pt', return_dict=True).to(model_qa.device)
        target_new_tokens_len = len(tok_qa.encode(label, add_special_tokens=False)) if label is not None else 16
        prompt_tok = tok_qa(prompt_qa, return_tensors="pt").to(model_qa.device)  # system_msg_qa+' '+
        gen_token = model_qa.generate(**prompt_tok, max_new_tokens=target_new_tokens_len, pad_token_id=tok_qa.eos_token_id, use_cache=False)
        output_text = gen_token.detach().cpu().numpy().tolist()[0][-target_new_tokens_len:]
        output_text = tok_qa.decode(output_text, skip_special_tokens=True)
        if label is None:  # For locality questions only return the output, do evaluation after the post-edit is collected in locality_acc_llm()
            return None, output_text
        return evaluate_response(hparams, model_eval, tok_eval, prompt_qa, output_text, label, device_eval)
    
    if isinstance(prompt_qa, list):
        for i, prompt in enumerate(prompt_qa):
            label_ = label[i] if label is not None else None
            return test_prediction_acc_single(hparams, model_qa, tok_qa, model_eval, tok_eval, device_eval, prompt, label_, system_msg)
    else:
        return test_prediction_acc_single(hparams, model_qa, tok_qa, model_eval, tok_eval, device_eval, prompt_qa, label, system_msg)


def test_prediction_acc_single(hparams, model_qa, tok_qa, model_eval, tok_eval, device_eval, prompt_qa, label, system_msg_qa):
    model_qa_name = hparams.model_name
    user_msg_qa = prompt_qa # f'Question: {prompt_qa}. Answer:'
    if 'llama' in model_qa_name.lower() or 'Mistral-7B-Instruct-v0.3' in model_qa_name:
        messages_qa = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": user_msg_qa}]
    elif 'gemma' in model_qa_name.lower():
        messages_qa = [{"role": "user", "content": system_msg_qa+' '+user_msg_qa}]
    elif 'vicuna' in model_qa_name.lower() or 'alpaca' in model_qa_name.lower():
        messages_qa = f"{system_msg_qa} Question: {user_msg_qa} Answer:"  # template for vicuna only
    else:
        messages_qa = [system_msg_qa+' '+user_msg_qa]

    output_qa = get_response(hparams, model_qa, tok_qa, messages_qa, max_new_tokens=16)  # , eval_flag=False, device_eval=device_eval
    # print(f'+++++ model_qa_name: {model_qa_name} +++++ user_msg_qa: {user_msg_qa} +++++ output_qa: {output_qa} +++++ system_msg_qa: {system_msg_qa}')

    if label is None:  # For locality questions only return the output, do evaluation after the post-edit is collected in locality_acc_llm()
        return None, output_qa
    
    return evaluate_response(hparams, model_eval, tok_eval, prompt_qa, output_qa, label, device_eval)


def test_prediction_acc_multi_turn(hparams, model_qa, tok_qa, model_eval, tok_eval, device_eval, prompt_qa, label, multi_turn, yes_question, no_question, pre_or_post, vanilla_generation=False):
    acc_ls = []
    output_qa_ls = []
    # questions_follow_up = ['Your answer is wrong. Are you going to change it? Respond with "Yes" or "No."' for _ in range(10)]
    # questions_follow_up = [f'Your answer is wrong. {no_question} Respond with "Yes" or "No."' for _ in range(10)]  # Yes old
    # # questions_follow_up = [f"The correct answer should be {label}, right? Answer Yes or No." for _ in range(10)]
    # label_follow_up = 'No'

    if multi_turn == 'sure':
        questions_follow_up = [f"Are you sure about your answer to the original question? Respond with 'Yes' or 'No.'" for _ in range(10)]
    elif multi_turn == 'yes':
        
        questions_follow_up = [f'Your answer to the original question is wrong. {yes_question} Respond with "Yes" or "No."' for _ in range(10)] 
    label_follow_up = 'Yes'

    messages_qa = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": prompt_qa}]
    if vanilla_generation and pre_or_post=='post':
        target_new_tokens_len = len(tok_qa.encode(label, add_special_tokens=False)) if label is not None else 16
        prompt_tok = tok_qa(prompt_qa, return_tensors="pt").to(model_qa.device)
        gen_token = model_qa.generate(**prompt_tok, max_new_tokens=target_new_tokens_len, pad_token_id=tok_qa.eos_token_id, use_cache=False)
        output_text = gen_token.detach().cpu().numpy().tolist()[0][-target_new_tokens_len:]
        current_output = tok_qa.decode(output_text, skip_special_tokens=True)
    else:
        current_output = get_response(hparams, model_qa, tok_qa, messages_qa, max_new_tokens=16)
    eval_acc, _ = evaluate_response(hparams, model_eval, tok_eval, prompt_qa, current_output, label, device_eval)
    acc_ls.append(eval_acc)
    output_qa_ls.append(current_output)

    for question in questions_follow_up:
        messages_qa.append({"role": "assistant", "content": current_output})
        messages_qa.append({"role": "user", "content": question})
        
        if vanilla_generation and pre_or_post=='post':
            target_new_tokens_len = len(tok_qa.encode(label, add_special_tokens=False)) if label is not None else 16
            formatted_input = tok_qa.apply_chat_template(messages_qa, tokenize=False, add_generation_prompt=True)
            prompt_tok = tok_qa(formatted_input, return_tensors="pt").to(model_qa.device)
            gen_token = model_qa.generate(**prompt_tok, max_new_tokens=target_new_tokens_len, pad_token_id=tok_qa.eos_token_id, use_cache=False)
            output_text = gen_token.detach().cpu().numpy().tolist()[0][-target_new_tokens_len:]
            current_output = tok_qa.decode(output_text, skip_special_tokens=True)
        else:
            current_output = get_response(hparams, model_qa, tok_qa, messages_qa, max_new_tokens=16)
        
        eval_acc, _ = evaluate_response(hparams, model_eval, tok_eval, prompt_qa, current_output, label_follow_up, device_eval)
        acc_ls.append(eval_acc)
        output_qa_ls.append(current_output)
        
    return acc_ls, output_qa_ls

        
def compute_edit_or_rephrase_quality(
    hparams: HyperParams,
    model,
    tok: AutoTokenizer,
    model_eval,
    tok_eval,
    device_eval,
    prompt: str,
    target_new: str,
    multi_turn: str,
    yes_question: str = None,
    no_question: str = None,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em',
    pre_or_post: str = 'pre'
) -> typing.Dict:
    if test_rephrase:
        key = 'rephrase'
    else:
        key = 'edit'
    if multi_turn is not None and key == 'edit':  # test multi-turn for the efficacy questions
        acc_ls, output_ls = test_prediction_acc_multi_turn(hparams, model, tok, model_eval, tok_eval, device_eval, prompt, target_new, multi_turn,
                                                           yes_question, no_question, pre_or_post, vanilla_generation=hparams.alg_name=='GRACE')
        return {f"{key}_acc": [acc_ls[0]], f"{key}_output": [output_ls[0]], f"{key}_acc_multi_turn": acc_ls, f"{key}_output_multi_turn": output_ls}
    else:
        acc, model_output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, prompt, target_new, 
                                                pre_or_post, vanilla_generation=hparams.alg_name=="GRACE")#
        return {f"{key}_acc": [acc], f"{key}_output": [model_output]}


def compute_general_quality(
    hparams: HyperParams,
    model,
    tok: AutoTokenizer,
    model_eval, 
    tok_eval, 
    device_eval,
    question_key: str,
    prompt: typing.Union[str, List[str]],
    question_ground_truth: typing.Union[str, List[str]],
    pre_or_post: str,
) -> typing.Dict:
    acc, model_output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, prompt, question_ground_truth, 
                                            pre_or_post, vanilla_generation=hparams.alg_name=='GRACE')
    return {f"{question_key}_acc": [acc], f"{question_key}_output": [model_output]}


def compute_multiple_choice_quality(hparams, model, tok, model_eval, tok_eval, device_eval, question_key, prompt_qa, label, pre_or_post):
    system_msg_multiple_choice = "Always respond to the multiple-choice question by selecting from the provided options. Only output the choice letter (A, B, C, or D)."
    acc, model_output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, prompt_qa, label, 
                                            pre_or_post, vanilla_generation=hparams.alg_name=='GRACE', system_msg=system_msg_multiple_choice)
    return {f"{question_key}_acc": [acc], f"{question_key}_output": [model_output]}


def compute_edit_quality(
    hparams: HyperParams,
    model,
    tok: AutoTokenizer,
    model_eval, 
    tok_eval, 
    device_eval,
    record: typing.Dict,
    multi_turn: str,
    eval_metric: str = 'token_em',
    test_generation = False,
    icl_pre_edit=True,
    pre_or_post='pre'
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired edit (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: dataset record
    :return: Dictionary containing rewriting metrics
    """
    if isinstance(model, LORA):
        model=model.model
    # First, unpack edit evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    edit_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None

    if hparams.alg_name in ['ICL', 'IKE'] and icl_pre_edit == False:
        icl_prompt = f"New Fact: Q: {edit_prompts} A: {target_new}\n"
    else:
        icl_prompt = ""

    
    yes_question = record['yes_questions']['yes']['prompt'] if 'yes_questions' in record.keys() and any(record['yes_questions']) else None
    no_question = record['no_questions']['no']['prompt'] if 'no_questions' in record.keys() and any(record['no_questions']) else None
    ret = compute_edit_or_rephrase_quality(hparams, model, tok, model_eval, tok_eval, device_eval, icl_prompt+edit_prompts, target_new, 
                                           multi_turn, yes_question, no_question, eval_metric=eval_metric, pre_or_post=pre_or_post)

    ret['locality'] = {}
    ret['portability'] = {}
    ret['yes_questions'] = {}
    ret['no_questions'] = {}
    ret['multiple_choice_questions'] = {}
    ret['reversed_relation_questions'] = {}
    ret['questions_2hop'] = {}
    ret['questions_3hop'] = {}
    ret['questions_4hop'] = {}
    ret['questions_5hop'] = {}
    ret['questions_6hop'] = {}
    ret['harm_original_text'] = {}

    if rephrase_prompts is not None:
        ret.update(
            compute_edit_or_rephrase_quality(hparams, model, tok, model_eval, tok_eval, device_eval, icl_prompt+rephrase_prompts, target_new, 
                                             multi_turn, test_rephrase=True, eval_metric=eval_metric, pre_or_post=pre_or_post)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            locality_prompt = record['locality'][locality_key]['prompt']
            if isinstance(locality_prompt, list):
                locality_prompt = [e+icl_prompt for e in locality_prompt]
            else:
                locality_prompt = icl_prompt + locality_prompt
            ret['locality'].update(
                compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, locality_key, locality_prompt, None, pre_or_post=pre_or_post)  # record['locality'][locality_key]['ground_truth'] ground_truth is not used in locality evaluation
            )

    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            portability_prompt = record['portability'][portability_key]['prompt']
            if isinstance(portability_prompt, list):
                portability_prompt = [e+icl_prompt for e in portability_prompt]
            else:
                portability_prompt = icl_prompt + portability_prompt
            ret['portability'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, portability_key, portability_prompt, record['portability'][portability_key]['ground_truth'], pre_or_post))
    
    if 'yes_questions' in record.keys() and any(record['yes_questions']):
        for key in record['yes_questions'].keys():
            yes_question = record['yes_questions'][key]['prompt']
            if isinstance(yes_question, list):
                yes_question = [e+icl_prompt for e in yes_question]
            else:
                yes_question = icl_prompt + yes_question
            ret['yes_questions'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, yes_question, record['yes_questions'][key]['ground_truth'], pre_or_post))

    if 'no_questions' in record.keys() and any(record['no_questions']):
        for key in record['no_questions'].keys():
            no_question = record['no_questions'][key]['prompt']
            if isinstance(no_question, list):
                no_question = [e+icl_prompt for e in no_question]
            else:
                no_question = icl_prompt + no_question
            ret['no_questions'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, no_question, record['no_questions'][key]['ground_truth'], pre_or_post))

    if 'multiple_choice_questions' in record.keys() and any(record['multiple_choice_questions']):
        for key in record['multiple_choice_questions'].keys():
            multiple_choice_question = record['multiple_choice_questions'][key]['prompt']
            if isinstance(multiple_choice_question, list):
                multiple_choice_question = [e+icl_prompt for e in multiple_choice_question]
            else:
                multiple_choice_question = icl_prompt + multiple_choice_question
            ret['multiple_choice_questions'].update(compute_multiple_choice_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, multiple_choice_question, record['multiple_choice_questions'][key]['ground_truth'], pre_or_post))

    if 'reversed_relation_questions' in record.keys() and any(record['reversed_relation_questions']):
        for key in record['reversed_relation_questions'].keys():
            reversed_relation_question = record['reversed_relation_questions'][key]['prompt']
            if isinstance(reversed_relation_question, list):
                reversed_relation_question = [e+icl_prompt for e in reversed_relation_question]
            else:
                reversed_relation_question = icl_prompt + reversed_relation_question
            ret['reversed_relation_questions'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, reversed_relation_question, record['reversed_relation_questions'][key]['ground_truth'], pre_or_post))

    if 'questions_2hop' in record.keys() and any(record['questions_2hop']):
        for key in record['questions_2hop'].keys():
            question = record['questions_2hop'][key]['prompt']
            if isinstance(question, list):
                question = [e+icl_prompt for e in question]
            else:
                question = icl_prompt + question
            ret['questions_2hop'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, question, record['questions_2hop'][key]['ground_truth'], pre_or_post))

    if 'questions_3hop' in record.keys() and any(record['questions_3hop']):
        for key in record['questions_3hop'].keys():
            question = record['questions_3hop'][key]['prompt']
            if isinstance(question, list):
                question = [e+icl_prompt for e in question]
            else:
                question = icl_prompt + question
            ret['questions_3hop'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, question, record['questions_3hop'][key]['ground_truth'], pre_or_post))

    if 'questions_4hop' in record.keys() and any(record['questions_4hop']):
        for key in record['questions_4hop'].keys():
            question = record['questions_4hop'][key]['prompt']
            if isinstance(question, list):
                question = [e+icl_prompt for e in question]
            else:
                question = icl_prompt + question
            ret['questions_4hop'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, question, record['questions_4hop'][key]['ground_truth'], pre_or_post))

    if 'questions_5hop' in record.keys() and any(record['questions_5hop']):
        for key in record['questions_5hop'].keys():
            question = record['questions_5hop'][key]['prompt']
            if isinstance(question, list):
                question = [e+icl_prompt for e in question]
            else:
                question = icl_prompt + question

            ret['questions_5hop'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, question, record['questions_5hop'][key]['ground_truth'], pre_or_post))

    if 'questions_6hop' in record.keys() and any(record['questions_6hop']):
        for key in record['questions_6hop'].keys():
            question = record['questions_6hop'][key]['prompt']
            if isinstance(question, list):
                question = [e+icl_prompt for e in question]
            else:
                question = icl_prompt + question
            ret['questions_6hop'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, question, record['questions_6hop'][key]['ground_truth'], pre_or_post)) 

    if test_generation:
        ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=edit_prompts if isinstance(edit_prompts,list) else [edit_prompts,], max_out_len=100, vanilla_generation=False)
    return ret


class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):

        assert hparams is not None, print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            if 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)  # GPT2Tokenizer
                self.tok.pad_token_id = self.tok.eos_token_id
            elif self.model_name in ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Meta-Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3'] and hparams.alg_name == 'ROME':
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower() or 'vicuna' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                print("WARNING: Probably Not Implemented") 
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id

            if self.tok is not None and (hparams.model_name=="EleutherAI/gpt-j-6b" or isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'

            if self.tok is not None and ('mistral' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT']): 
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             yes_questions: Optional[Dict] = None,
             no_questions: Optional[Dict] = None,
             locality_inputs: Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             multiple_choice_questions: Optional[Dict] = None,
             reversed_relation_questions: Optional[Dict] = None,
             questions_2hop: Optional[Dict] = None,
             questions_3hop: Optional[Dict] = None,
             questions_4hop: Optional[Dict] = None,
             questions_5hop: Optional[Dict] = None,
             questions_6hop: Optional[Dict] = None,
             harm_original_text: Optional[Union[str, List[str]]] = None,
             keep_original_weight=False,
             verbose=True,
             summary_metrics=False, 
             eval_model_id='meta-llama/Meta-Llama-3.1-8B-Instruct',
             device_eval='cuda:0',
             multi_turn=None,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `locality_inputs`: dict
            for locality
        """
        ## Model for evaluating the correctness of the prediction compared to the label
        model_eval = AutoModelForCausalLM.from_pretrained(eval_model_id, torch_dtype='auto').to(device_eval)
        tok_eval = AutoTokenizer.from_pretrained(eval_model_id)

        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        # assert (locality_prompts is None and locality_ground_truth is None) or \
        #        (isinstance(locality_prompts, str) and isinstance(locality_ground_truth, str)) or \
        #        len(locality_prompts) == len(locality_ground_truth) or print('Error in locality Input.')
        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = self._prepare_requests(prompts, target_new, ground_truth, rephrase_prompts, yes_questions, no_questions, 
                                              locality_inputs, portability_inputs, multiple_choice_questions, reversed_relation_questions,
                                              questions_2hop, questions_3hop, questions_4hop, questions_5hop, questions_6hop, harm_original_text, **kwargs)
        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1, print(f'Single Edit, pls set the batch_size to 1....')

        # if not os.path.exists(RESULTS_DIR):
        #     os.mkdir(RESULTS_DIR)
        # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
        # if not os.path.exists(base_case_path):
        #     os.mkdir(base_case_path)
        # print(f"Results will be stored at {base_case_path}")

        if self.alg_name == 'FT-Api':
            all_metrics = []
            for i, request in enumerate(requests):
                metrics = {
                    "pre": {}
                }
                all_metrics.append(metrics)

            start = time()
            edited_model, weights_copy = self.apply_algo(
                requests,
                self.hparams
            )
            exec_time = time() - start

            LOG.info(f"Execution editing took {exec_time}")

            for i, request in enumerate(requests):
                all_metrics[i].update({
                    'case_id': i,
                    "requested_edit": request,
                    "time": exec_time,
                    "post": {}
                })

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            return all_metrics, edited_model, weights_copy

        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in enumerate(tqdm(requests)):
                if self.alg_name in ['IKE', 'ICL']:
                    # assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    metrics = {
                        # "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                        #                                 request, self.hparams.device, pre_edit=True)
                        "pre": compute_edit_quality(self.hparams, self.model, self.tok, model_eval, tok_eval, device_eval, request, multi_turn,
                                                    test_generation=test_generation, icl_pre_edit=True, pre_or_post='pre')
                    }
                else:
                    metrics = {
                        "pre": compute_edit_quality(self.hparams, self.model, self.tok, model_eval, tok_eval, device_eval, request, multi_turn,
                                                    test_generation=test_generation, pre_or_post='pre')
                    }
                all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                ### Store the pre_edit metric to refrain computing repeatedly
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

        for i, request in enumerate(requests):
            start = time()

            if self.alg_name in ['IKE', 'ICL']:
                edited_model, weights_copy = self.model, {}
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    # train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")

            start = time()
            if self.alg_name in ['IKE', 'ICL']:
                all_metrics[i].update({
                    'case_id': i,
                    "requested_edit": request,
                    "time": exec_time,
                    "post": compute_edit_quality(self.hparams, edited_model, self.tok, model_eval, tok_eval, device_eval, request, multi_turn,
                                                 test_generation=test_generation, icl_pre_edit=False, pre_or_post='post'),
                })
            else:
                all_metrics[i].update({
                    'case_id': i,
                    "requested_edit": request,
                    "time": exec_time,
                    "post": compute_edit_quality(self.hparams, edited_model, self.tok, model_eval, tok_eval, device_eval, request, multi_turn,
                                                 test_generation=test_generation, pre_or_post='post'),
                })
            if "metric_kwargs" in kwargs:
                all_metrics[i].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok, metric_kwargs=kwargs["metric_kwargs"][i], device=self.hparams.device))
            if self.alg_name == 'KN' or (self.alg_name == 'GRACE' and keep_original_weight):
                with torch.no_grad():
                    weights_copy() # unpatch_fn
            elif self.alg_name == 'LoRA' and keep_original_weight:
                edited_model.unload()
                del self.model.peft_config
            elif self.alg_name == 'MELO':
                self.model = edited_model
            elif self.alg_name == 'LoRA' and not keep_original_weight:
                self.model = edited_model
            else:
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            if 'locality' in all_metrics[i]['post'].keys():
                for locality_key in request['locality'].keys():
                    locality_result = []
                    for question, pre_edit_output, post_edit_output in zip(all_metrics[i]['requested_edit']['locality']['locality']['prompt'], 
                                                                 all_metrics[i]['pre']['locality'][f'{locality_key}_output'], 
                                                                 all_metrics[i]['post']['locality'][f'{locality_key}_output']):
                        acc, _ = evaluate_response(self.hparams, model_eval, tok_eval, question, pre_edit_output, post_edit_output, device_eval)
                        locality_result.append(acc)
                        # locality_result.append(locality_acc_llm(self.hparams, pre_edit_output, post_edit_output))
                    all_metrics[i]['post']['locality'][f'{locality_key}_acc'] = locality_result
                    all_metrics[i]['pre']['locality'].pop(f'{locality_key}_acc')
                    

            LOG.info(f"Evaluation took {time() - start}")

            if verbose:
                LOG.info(
                    f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                )
            # case_result_path = base_case_path / f"case_{i}.json"

            # Dump metrics in .json
            # with open(case_result_path, "w") as f:
            #     json.dump(metrics, f, indent=1)

        if isinstance(edited_model, LORA):
            edited_model=edited_model.model
        #for melo
        
        if summary_metrics and len(all_metrics)!=0:
            if isinstance(all_metrics, dict):
                all_metrics = [all_metrics,]
            # logs_dir = './logs'  
            # if not os.path.exists(logs_dir):  
            #     os.makedirs(logs_dir)  
            # output_file = os.path.join(logs_dir, 'results.json')
            # with open(output_file, 'w') as f:  
            #     json.dump(all_metrics, f, ensure_ascii=False, indent=4)
            
            mean_metrics = dict()
            for eval in ["pre", "post"]:
                mean_metrics[eval] = dict()
                for key in ["edit_acc", "rephrase_acc"]:
                    if key in all_metrics[0][eval].keys():
                        mean_metrics[eval][key] = np.mean([metric[eval][key] for metric in all_metrics])
                for key in ["locality", "portability", "yes_questions", "no_questions", "multiple_choice_questions", "reversed_relation_questions", 
                            "questions_2hop", "questions_3hop", "questions_4hop", "questions_5hop", "questions_6hop"]:
                    if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                        mean_metrics[eval][key] = dict()
                        # for lkey in all_metrics[0][eval][key].keys():
                        #     if lkey.endswith("acc"):
                        #         mean_metrics[eval][key][lkey] = np.mean([metric[eval][key][lkey] for metric in all_metrics])
                        for lkey in get_all_acc_keys(all_metrics):
                            metrics = [metric[eval][key][lkey] for metric in all_metrics if lkey in metric[eval][key].keys()]
                            if len(metrics) > 0:
                                mean_metrics[eval][key][lkey] = np.mean(metrics)
            mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])
            
            print("Metrics Summary: ", mean_metrics)


        # del model_eval
        return all_metrics, edited_model, weights_copy


    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]
        
    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          target_new: Union[str, List[str]],
                          ground_truth: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          yes_questions: Optional[Dict] = None,
                          no_questions: Optional[Dict] = None,
                          locality_inputs: Optional[Dict] = None,
                          portability_inputs: Optional[Dict] = None,
                          multiple_choice_questions: Optional[Dict] = None,
                          reversed_relation_questions: Optional[Dict] = None,
                          questions_2hop: Optional[Dict] = None,
                          questions_3hop: Optional[Dict] = None,
                          questions_4hop: Optional[Dict] = None,
                          questions_5hop: Optional[Dict] = None,
                          questions_6hop: Optional[Dict] = None,
                          harm_original_text: Union[str, List[str]] = None,
                          **kwargs
                          ):

        requests = [{
            'prompt': prompt,
            'target_new': target_new_,
            'ground_truth': ground_truth_,
            'portability': {},
            'locality': {},
            'yes_questions': {},
            'no_questions': {},
            'multiple_choice_questions': {},
            'reversed_relation_questions': {},
            'questions_2hop': {},
            'questions_3hop': {},
            'questions_4hop': {},
            'questions_5hop': {},
            'questions_6hop': {},
            'harm_original_text': {}
        }
        for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
        ]

        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )

        if harm_original_text is not None:
            if isinstance(harm_original_text, str):
                harm_original_text = [harm_original_text,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'harm_original_text': harm_original_text[i],
                    }
                )

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': rephrase_prompts[i],
                    }
                )
        if locality_inputs is not None:
            for locality_key in locality_inputs.keys():
                if isinstance(locality_inputs[locality_key]['prompt'], str):
                    locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                #     locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
                # assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
                # == len(requests), print('One Edit instance needs one locality input.....')

                for i, request in enumerate(requests):
                    if locality_inputs[locality_key]['prompt'][i] is not None:
                        request['locality'].update(
                            {
                                locality_key: {
                                    f'prompt': locality_inputs[locality_key]['prompt'][i],
                                    # f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                                }
                            }
                        )

        if portability_inputs is not None:
            for portability_key in portability_inputs.keys():
                if isinstance(portability_inputs[portability_key]['prompt'], str):
                    portability_inputs[portability_key]['prompt'] = [portability_inputs[portability_key]['prompt'],]
                    portability_inputs[portability_key]['ground_truth'] = [portability_inputs[portability_key]['ground_truth'], ]
                assert len(portability_inputs[portability_key]['prompt']) == len(portability_inputs[portability_key]['ground_truth']) \
                == len(requests), print('One Edit instance needs one portability input.....')

                for i, request in enumerate(requests):
                    if portability_inputs[portability_key]['prompt'][i] is not None:
                        request['portability'].update(
                            {
                                portability_key: {
                                    'prompt': portability_inputs[portability_key]['prompt'][i],
                                    'ground_truth': portability_inputs[portability_key]['ground_truth'][i]
                                }
                            }
                        )

        if yes_questions is not None:
            for key in yes_questions.keys():
                if isinstance(yes_questions[key]['prompt'], str):
                    yes_questions[key]['prompt'] = [yes_questions[key]['prompt'],]
                    yes_questions[key]['ground_truth'] = [yes_questions[key]['ground_truth'], ]
                assert len(yes_questions[key]['prompt']) == len(yes_questions[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if yes_questions[key]['prompt'][i] is not None:
                        request['yes_questions'].update({key: {'prompt': yes_questions[key]['prompt'][i], 'ground_truth': yes_questions[key]['ground_truth'][i]}})

        if no_questions is not None:
            for key in no_questions.keys():
                if isinstance(no_questions[key]['prompt'], str):
                    no_questions[key]['prompt'] = [no_questions[key]['prompt'],]
                    no_questions[key]['ground_truth'] = [no_questions[key]['ground_truth'], ]
                assert len(no_questions[key]['prompt']) == len(no_questions[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if no_questions[key]['prompt'][i] is not None:
                        request['no_questions'].update({key: {'prompt': no_questions[key]['prompt'][i],  'ground_truth': no_questions[key]['ground_truth'][i]}})

        if multiple_choice_questions is not None:
            for key in multiple_choice_questions.keys():
                if isinstance(multiple_choice_questions[key]['prompt'], str):
                    multiple_choice_questions[key]['prompt'] = [multiple_choice_questions[key]['prompt'],]
                    multiple_choice_questions[key]['ground_truth'] = [multiple_choice_questions[key]['ground_truth'], ]
                assert len(multiple_choice_questions[key]['prompt']) == len(multiple_choice_questions[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if multiple_choice_questions[key]['prompt'][i] is not None:
                        request['multiple_choice_questions'].update({key: {'prompt': multiple_choice_questions[key]['prompt'][i], 'ground_truth': multiple_choice_questions[key]['ground_truth'][i]}})

        if reversed_relation_questions is not None:
            for key in reversed_relation_questions.keys():
                if isinstance(reversed_relation_questions[key]['prompt'], str):
                    reversed_relation_questions[key]['prompt'] = [reversed_relation_questions[key]['prompt'],]
                    reversed_relation_questions[key]['ground_truth'] = [reversed_relation_questions[key]['ground_truth'], ]
                assert len(reversed_relation_questions[key]['prompt']) == len(reversed_relation_questions[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if reversed_relation_questions[key]['prompt'][i] is not None:
                        request['reversed_relation_questions'].update({key: {'prompt': reversed_relation_questions[key]['prompt'][i], 'ground_truth': reversed_relation_questions[key]['ground_truth'][i]}})

        if questions_2hop is not None:
            for key in questions_2hop.keys():
                if isinstance(questions_2hop[key]['prompt'], str):
                    questions_2hop[key]['prompt'] = [questions_2hop[key]['prompt'],]
                    questions_2hop[key]['ground_truth'] = [questions_2hop[key]['ground_truth'], ]
                assert len(questions_2hop[key]['prompt']) == len(questions_2hop[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if questions_2hop[key]['prompt'][i] is not None:
                        request['questions_2hop'].update({key: {'prompt': questions_2hop[key]['prompt'][i], 'ground_truth': questions_2hop[key]['ground_truth'][i]}})

        if questions_3hop is not None:
            for key in questions_3hop.keys():
                if isinstance(questions_3hop[key]['prompt'], str):
                    questions_3hop[key]['prompt'] = [questions_3hop[key]['prompt'],]
                    questions_3hop[key]['ground_truth'] = [questions_3hop[key]['ground_truth'], ]
                assert len(questions_3hop[key]['prompt']) == len(questions_3hop[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if questions_3hop[key]['prompt'][i] is not None:
                        request['questions_3hop'].update({key: {'prompt': questions_3hop[key]['prompt'][i], 'ground_truth': questions_3hop[key]['ground_truth'][i]}})

        if questions_4hop is not None:
            for key in questions_4hop.keys():
                if isinstance(questions_4hop[key]['prompt'], str):
                    questions_4hop[key]['prompt'] = [questions_4hop[key]['prompt'],]
                    questions_4hop[key]['ground_truth'] = [questions_4hop[key]['ground_truth'], ]
                assert len(questions_4hop[key]['prompt']) == len(questions_4hop[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if questions_4hop[key]['prompt'][i] is not None:
                        request['questions_4hop'].update({key: {'prompt': questions_4hop[key]['prompt'][i], 'ground_truth': questions_4hop[key]['ground_truth'][i]}})

        if questions_5hop is not None:
            for key in questions_5hop.keys():
                if isinstance(questions_5hop[key]['prompt'], str):
                    questions_5hop[key]['prompt'] = [questions_5hop[key]['prompt'],]
                    questions_5hop[key]['ground_truth'] = [questions_5hop[key]['ground_truth'], ]
                assert len(questions_5hop[key]['prompt']) == len(questions_5hop[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if questions_5hop[key]['prompt'][i] is not None:
                        request['questions_5hop'].update({key: {'prompt': questions_5hop[key]['prompt'][i], 'ground_truth': questions_5hop[key]['ground_truth'][i]}})

        if questions_6hop is not None:
            for key in questions_6hop.keys():
                if isinstance(questions_6hop[key]['prompt'], str):
                    questions_6hop[key]['prompt'] = [questions_6hop[key]['prompt'],]
                    questions_6hop[key]['ground_truth'] = [questions_6hop[key]['ground_truth'], ]
                assert len(questions_6hop[key]['prompt']) == len(questions_6hop[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if questions_6hop[key]['prompt'][i] is not None:
                        request['questions_6hop'].update({key: {'prompt': questions_6hop[key]['prompt'][i], 'ground_truth': questions_6hop[key]['ground_truth'][i]}})
        return requests


    def normal_edit(
        self,
        prompts: List[str],
        target_new: List[str],
        keep_original_weight=False,
        epoch: int=5,
    ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        assert BatchEditor.is_batchable_method(self.alg_name), print(f'The Method {self.alg_name} can not batch edit examples.')

        requests = self._prepare_requests(prompts, target_new, ground_truth)

        assert hasattr(self.hparams, 'batch_size'), print(f'Method {self.alg_name} found, pls specify the batch_size....')

        # print(f"[editor.py][batch_edit] `batch_size`={self.hparams.batch_size}")
        # for epc in range(epoch):
        #     print(f"[editor.py][batch_edit] `Epoch` = {epc+1}")
        #     for record_chunks in self._chunks(requests, self.hparams.batch_size):
        start = time()

        edited_model, weights_copy = self.apply_algo(
            self.model,
            self.tok,
            requests,  # record_chunks -> requests
            self.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=keep_original_weight,
        )
        exec_time = time() - start
        LOG.info(f"Execution editing took {exec_time}")

        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        return None, edited_model, weights_copy

