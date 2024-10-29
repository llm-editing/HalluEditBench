# Can Knowledge Editing Really Correct Hallucinations?

- **Respository Oveview**: This repository contains the code, results and dataset for the paper **["Can Knowledge Editing Really Correct Hallucinations?"](https://arxiv.org/abs/2410.16251)**
- **TLDR**: We proposed HalluEditBench to holistically benchmark knowledge editing methods in correcting real-world hallucinations on five dimensions including Efficacy, Generalization, Portability, Locality, and Robustness. We find that their effectiveness could be far from what their performance on existing datasets suggests, and the performance beyond Efficacy for all methods is generally unsatisfactory.
- **Authors** : [Baixiang Huang\*](https://baixianghuang.github.io/), [Canyu Chen\*](https://canyuchen.com), [Xiongxiao Xu](https://xiongxiaoxu.github.io/), [Ali Payani](https://www.linkedin.com/in/ali-payani-59267515/), [Kai Shu](https://www.cs.emory.edu/~kshu5/) (*equal contributions)
- **Correspondence to**: Baixiang Huang <<bhuang15@hawk.iit.edu>>, Canyu Chen <<cchen151@hawk.iit.edu>>, Kai Shu <<kai.shu@emory.edu>>.
- **Paper** : [Read our paper](https://arxiv.org/abs/2410.16251)
- **Project Website**: Visit the project website [https://llm-editing.github.io](https://llm-editing.github.io/) for more resources.


## Overview
Large Language Models (LLMs) suffer from hallucinations, referring to the non-factual information in generated content, despite their superior capacities across tasks. Meanwhile, knowledge editing has been developed as a new popular paradigm to correct the erroneous factual knowledge encoded in LLMs with the advantage of avoiding retraining from scratch. However, one common issue of existing evaluation datasets for knowledge editing is that they do not ensure LLMs actually generate hallucinated answers to the evaluation questions before editing. When LLMs are evaluated on such datasets after being edited by different techniques, it is hard to directly adopt the performance to assess the effectiveness of different knowledge editing methods in correcting hallucinations. Thus, the fundamental question remains insufficiently validated: Can knowledge editing really correct hallucinations in LLMs? 

We proposed **HalluEditBench** to holistically benchmark knowledge editing methods in correcting real-world hallucinations. First, we rigorously construct a massive hallucination dataset with 9 domains, 26 topics and more than 6,000 hallucinations. Then, we assess the performance of knowledge editing methods in a holistic way on five dimensions including ***Efficacy***, ***Generalization***, ***Portability***, ***Locality***, and ***Robustness***. Through **HalluEditBench**, we have provided new insights into the potentials and limitations of different knowledge editing methods in correcting hallucinations, which could inspire future improvements and facilitate the progress in the field of knowledge editing.


<img src="data/intro.jpg" width=100%>


# Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
    1. [Data Preparation](#data-preparation)
    2. [Running Experiments](#running-experiments)
<!-- 5. [Contributing](#contributing) -->
5. [Acknowledgements](#acknowledgements)


## Repository Structure
- `data/`: Contains the hallucination detection dataset.
- `code/`: Includes scripts and code to evaluate hallucination mitigation using knowledge editing methods (and reproduce the results in the paper).
- `results/`: Results of the experiments that we report in the paper.


## Installation
To set up the environment for running the code, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/link-omitted-during-review/hallu-edit.git
    cd hallu-edit
    ```

2. Create a virtual environment and activate it:
    ```bash
    conda create -n HalluEdit python=3.9
    conda activate HalluEdit
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Usage

### Data Preparation

1. Datasets are stored in the `data/` directory. There are three folders: 

```bash
data/
    ├── questions
    │   └── hallucination_final
    │       ├── llama_2_7b_chat_hf
    │       ├── meta_llama_3_8b_instruct
    │       └── mistral_7b_instruct_v0.3
    ├── topic
    └── triplet
```
`questions` contains the pre-processed hallucination detection dataset, including the questions we used to evaluate the editing methods. `topic` contains the topics we selected from WikiData, and `triplet` contains the raw knowledge triplets that were used to generate the questions for hallucination detection.

### Running Experiments

**Run example**: To get started (e.g. using ROME to edit llama3-8b on the places_landmark data), run:

```bash
cd ./code
python3 edit_all_method.py \
    --model_name=llama3-8b \
    --edit_method=ROME \
    --topic_name=places_landmark \
    --device_edit=0 \
    --device_eval=1 \
    --data_size=5 \
    --results_dir=../new_results_dir \
    --question_types rephrase_questions questions_2hop
```

Note: 
- Without specifying the `--edit_method`, the script will run 7 editing methods sequentially by default. 
- Specify `--question_types` to choose specific types of questions in the evaluation (The example above will only evalute 2-hop questions and rephrased questions). Otherwise, the script will run all the question types (yes_questions, no_questions, locality_questions, rephrase_questions, multiple_choice_questions, reversed_relation_questions, questions_2hop, questions_3hop, questions_4hop, questions_5hop, questions_6hop). The original questions is always included.
- Specify `--results_dir` to save the results to a specific directory, otherwise the default directory is where we save the results that we report in the paper. You can also use `--overwrite_result` to overwrite the existing result file.
<!-- If you use an API model (such as GPT-4) as the evaluator, you need to set your `YOUR_API_KEY` in Line 60 of `code/editor_new_eval.py`. One example is as follows: -->

To run the multi-turn editing, here is an example:
```bash
python3 edit_all_method_multi_turn.py \
    --model_name=llama3-8b \
    --edit_method=ROME \
    --topic_name=places_landmark \
    --device_edit=0 \
    --device_eval=1 \
    --model_eval=meta-llama/Meta-Llama-3-8B-Instruct \
    --data_size=5 \
    --results_dir=../new_results_dir \
    --multi_turn=yes \
    --multi_turn_num=10
```
- Use `--multi_turn` to choose the type of multi-turn evaluation (`yes` or `sure`).
- Use `--multi_turn_num` to set the number of turns for multi-turn evaluation.


We use a local LLM (e.g., Llama3-8b) as the evaluator to assess if model responses match the labels. For experiments, we recommend using at least one GPU with 48 GB of memory (e.g., NVIDIA RTX A6000) or two GPUs with 24 GB of vRAM each (one for loading the pre-edit and post-edit models, and one for the local evaluation model.) Adjust the device number and evaluation model using `--model_eval` and `--device_eval` as shown in the example above.

For full experiments to reproduce the results in the paper:
1. Experiment for all the 26 topics:
    ```bash
    ./edit_all_topic.sh
    ```

2. Experiment for the robustness evaluation:
    ```bash
    ./code/edit_all_topic_multi_turn.sh
    ```


<!-- An OpenAI API key is required for GPT-4 evaluation. Save it in the "api_key.json" file. -->

We evaluate instruction-tuned models including `Llama-2-7B-chat`, `Llama-3-8B-Instruct`, and `Mistral-7B-v0.3`. All parameters are in the `code/hparams/<method_name>/<model_name>`. 

Results are stored at `llama_2_7b_chat_hf`, `meta_llama_3_8b_instruct`, `mistral_7b_instruct_v0.3` under the `results` folder.

To summarize the results, use the jupyter notebook `code/result_table.ipynb`

<!-- 
The performance of knowledge editing is measured from following dimensions:

- `Efficacy`: whether the edited models could recall the exact editing fact under editing prompts
- `Generalization`: whether the edited models could recall the editing fact under paraphrase prompts
- `Locality`: whether the output of the edited models for inputs out of editing scope remains unchanged after editing
- `Additivity`: the degree of perturbation to neighboring knowledge when appending. -->


<!-- ## Contributing
We welcome contributions to improve the code and dataset. Please open an issue or submit a pull request if you have any suggestions or improvements. -->


<!-- ## License
This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). 


## Ethics Statement -->


## Acknowledgements
We gratefully acknowledge the use of code and data from the following projects: [GRACE](https://github.com/thartvigsen/grace), [EasyEdit](https://github.com/zjunlp/EasyEdit), [ROME](https://github.com/kmeng01/rome), [MEMIT](https://github.com/kmeng01/memit)
<!-- [IKE]() -->

## Citation
If you find our paper or code useful, we will greatly appreacite it if you could consider citing our paper:
```
@article{huang2024canknowledge,
    title   = {Can Knowledge Editing Really Correct Hallucinations?},
    author  = {Baixiang Huang and Canyu Chen and Xiongxiao Xu and Ali Payani and Kai Shu},
    year    = {2024},
    journal = {arXiv preprint arXiv: 2410.16251}
}
```

<!-- Please note that we do not have ownership of the data and therefore cannot provide a license or control its use. However, we kindly request that the data only be used for research purposes. -->
