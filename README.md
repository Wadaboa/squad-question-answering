# SQuAD Q&A

<p align="center">
  <img src="assets/img/squad-logo.jpg" alt="SQuAD logo"/>
</p>

This repository contains solutions to the question answering problem on the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) v1.1 dataset, which consists on selecting a possible answer to the given question as a span of words in the given context paragraph. The newest version (v2.0) of the dataset also contains unansweable questions, but the one on which we worked on (v1.1) does not.

## Installation

In order to install all the dependencies required by the project, you have two options:
1. Using `pip`: make sure that you `Python 3.8` installed on your system and run

```bash
python3 -m venv squad
source squad/bin/activate
pip install -r init/requirements.txt
```

2. Using `conda`: simply run the following command

```bash
conda env create --name squad -f init/environment.yml
conda activate squad
```

## Execution

### Training

The training part of the project is managed through a Jupyter notebook, in which you can select which model to train and which hyperparameters to use. 

Training and evaluation metrics, along with model checkpoints and results, are directly logged into a [W&B](https://wandb.ai) project, which is openly accessible [here](https://wandb.ai/wadaboa/squad-qa). Logging abilities are only granted to members of the team, so that if you want to launch your training run, you would have to disable `wandb`, by setting the environment variable `WANDB_DISABLED` to an empty value at the top of the notebook (`%env WANDB_DISABLED=`).

### Testing

The testing part of the project is managed using two Python scripts: 

1. `compute_answers.py`: given the path to the testing JSON file (formatted as the official SQuAD training set JSON), computes and saves another JSON file with the following format

```json
{
    "question_id": "textual answer",
    ...
}
```

2. `evaluate.py`: given the path to the same testing JSON file used in the `compute_answers.py` script and the JSON file produced by the script itself, prints to the standard output a dictionary of metrics such as the `F1` and `Exact Match` scores, which can be used to assess the performance of a trained model as done in the official SQuAD competition