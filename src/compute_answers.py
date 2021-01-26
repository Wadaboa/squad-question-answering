import argparse
import os

import numpy as np
import torch

import dataset
import tokenizer
import model
import training
import utils


MODELS = {
    "baseline": model.QABaselineModel,
    "bidaf": model.QABiDAFModel,
    "bert": model.QABertModel,
}
TOKENIZERS = {
    "baseline": tokenizer.get_standard_tokenizer,
    "bidaf": tokenizer.get_standard_tokenizer,
    "bert": tokenizer.get_bert_tokenizer,
}


def main(args):
    squad_dataset = dataset.SquadDataset(test_set_path=args.path)
    squad_tokenizer = TOKENIZERS[args.model](device=args.device)
    data_manager = dataset.SquadDataManager(
        squad_dataset, squad_tokenizer, device=args.device
    )

    torch_model = MODELS[args.model](device=args.device)
    torch_model.load_state_dict(torch.load(args.weights, map_location=args.device))
    print(f"Weights loaded from {args.weights}")

    trainer_args = utils.get_default_trainer_args()(
        output_dir="./checkpoints",
        per_device_eval_batch_size=args.batch,
        no_cuda=(args.device == "cpu"),
    )
    trainer = training.SquadTrainer(
        model=torch_model, args=trainer_args, data_collator=data_manager.tokenizer,
    )

    test_output = trainer.predict(data_manager.test_dataset)
    utils.save_answers(args.results, test_output.predictions[-1])
    print(f"Results saved in {args.results}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the testing set .json file")
    parser.add_argument(
        "-d",
        "--device",
        default=utils.get_device(),
        help="which device to use (defaults to GPU, if available)",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=MODELS.keys(),
        default="bert",
        help="model type to test",
    )
    parser.add_argument(
        "-w", "--weights", help="path to the model checkpoint to load",
    )
    parser.add_argument(
        "-r",
        "--results",
        default=f"{os.getcwd()}/results.json",
        help="where to save computed predictions",
    )
    parser.add_argument(
        "-b", "--batch", default=16, help="batch size for the test dataloader"
    )
    args = parser.parse_args()
    if args.weights is None:
        args.weights = f"results/models/{args.model}.bin"
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
