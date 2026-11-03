from pathlib import Path
import json
root=Path(__file__).parent
src=(root/'run_experiment.py').read_text()
cells=[]
def add(kind,text):
    c={'cell_type':kind,'metadata':{},'source':text.splitlines(True)}
    if kind=='code':c.update(execution_count=None,outputs=[])
    cells.append(c)
add('markdown','''# Airline Satisfaction — Experiment 1: pretrained TabICLv2\n\nDirect inference using frozen TabICLv2 classifier weights from `jingang/TabICL`, compared with a locally trained LightGBM control. No public submission files or target labels are reused.\n\nAttach **Predicting Airline Satisfaction (playground-series-s6e10)**, enable Internet for the first checkpoint download, and run cells in order. This CPU recipe matches the local experiment and may take tens of minutes. Outputs include validation metrics, separate model predictions, and `submission_selected.csv`. It does not submit automatically.\n\nValidation: 599,635 rows form the base-model training pool, 50,000 select among five fixed blend weights, and another 50,000 report the selected recipe. LightGBM early stopping uses a further split inside the training pool. TabICL uses 16,384 context examples; it is not trained from scratch or fine-tuned. This is a single holdout experiment, not cross-validation.\n\nSources and attribution:\n- TabICLv2 official implementation: https://github.com/soda-inria/tabicl\n- Official pretrained checkpoint (BSD-3-Clause): https://huggingface.co/jingang/TabICL\n- Competition: https://www.kaggle.com/competitions/playground-series-s6e10\n- Qu et al. (2026), TabICLv2: https://arxiv.org/abs/2602.11139\n''')
add('code','%pip install -q tabicl==2.2.0 lightgbm\n')
add('code',"from pathlib import Path\nfrom huggingface_hub import hf_hub_download\ncheckpoint = hf_hub_download(repo_id='jingang/TabICL', filename='tabicl-classifier-v2-20260212.ckpt')\npaths=[p.parent for p in Path('/kaggle/input').rglob('train.csv') if 'playground-series-s6e10' in str(p) and (p.parent/'test.csv').exists()]\nif len(paths)!=1: raise ValueError(f'Attach exactly one copy of the competition data: {paths}')\nDATA_DIR=paths[0]\n")
add('code','source = '+repr(src)+'\nPath("run_experiment.py").write_text(source)\n')
add('code',"from run_experiment import run\nmetrics=run(DATA_DIR,Path('/kaggle/working/experiment1'),checkpoint)\nmetrics['report']\n")
add('code',"import pandas as pd\nfrom IPython.display import display, FileLink\nsubmission_path=Path('/kaggle/working/experiment1/submission_selected.csv')\ndisplay(pd.read_csv(submission_path).head())\ndisplay(FileLink(str(submission_path)))\n")
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
(root/'airline_experiment1.ipynb').write_text(json.dumps(nb,indent=1))
