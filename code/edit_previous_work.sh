// ZsRE
python run_knowedit_llama2.py --editing_method=FT --hparams_dir=./hparams/FT-M/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre'

python run_knowedit_llama2.py --editing_method=FT-L --hparams_dir=./hparams/FT-L/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre'

python run_knowedit_llama2.py --editing_method=IKE --hparams_dir=./hparams/IKE/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre' --train_data_path=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json

python run_knowedit_llama2.py --editing_method=MEMIT --hparams_dir=./hparams/MEMIT/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre'
    
python run_knowedit_llama2.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre'

python run_knowedit_llama2.py --editing_method=LoRA --hparams_dir=./hparams/LoRA/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre'


// WikiRecent
python run_knowedit_llama2.py --editing_method=FT --hparams_dir=./hparams/FT-M/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent'

python run_knowedit_llama2.py --editing_method=FT-L --hparams_dir=./hparams/FT-L/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent'

python run_knowedit_llama2.py --editing_method=IKE --hparams_dir=./hparams/IKE/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent' --train_data_path=../data/KnowEdit/benchmark/wiki_recent/recent_test.json

python run_knowedit_llama2.py --editing_method=MEMIT --hparams_dir=./hparams/MEMIT/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent'
    
python run_knowedit_llama2.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent'

python run_knowedit_llama2.py --editing_method=LoRA --hparams_dir=./hparams/LoRA/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent'


// Wikibio
python run_knowedit_llama2.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio'

python run_knowedit_llama2.py --editing_method=FT --hparams_dir=./hparams/FT-M/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio'

python run_knowedit_llama2.py --editing_method=FT-L --hparams_dir=./hparams/FT-L/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio'

python run_knowedit_llama2.py --editing_method=MEMIT --hparams_dir=./hparams/MEMIT/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio'

python run_knowedit_llama2.py --editing_method=IKE --hparams_dir=./hparams/IKE/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio' --train_data_path=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json

python run_knowedit_llama2.py --editing_method=LoRA --hparams_dir=./hparams/LoRA/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio'


// Counter fact
python run_knowedit_llama2.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json --datatype='counterfact'

python run_knowedit_llama2.py --editing_method=MEMIT --hparams_dir=./hparams/MEMIT/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json --datatype='counterfact'

python run_knowedit_llama2.py --editing_method=FT --hparams_dir=./hparams/FT-M/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json --datatype='counterfact'

python run_knowedit_llama2.py --editing_method=FT-L --hparams_dir=./hparams/FT-L/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json --datatype='counterfact'

python run_knowedit_llama2.py --editing_method=IKE --hparams_dir=./hparams/IKE/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json --datatype='counterfact' --train_data_path=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json

python run_knowedit_llama2.py --editing_method=LoRA --hparams_dir=./hparams/LoRA/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json --datatype='counterfact'





