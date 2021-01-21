import argparse
import os

import dataset
import tokenizer
import model
import training
import utils


def main(args):
    squad_dataset = dataset.SquadDataset(test_set_path=args.path)
    bert_tokenizer = tokenizer.get_bert_tokenizer(device=args.device)
    bert_dm = dataset.SquadDataManager(
        squad_dataset, bert_tokenizer, device=args.device
    )

    bert_model = model.QABertModel(device=args.device)
    if args.weights:
        bert_model.load_state_dict(torch.load(args.weights, map_location=args.device))
        print(f"Weights loaded from {args.weights}")

    trainer_args = utils.get_default_trainer_args()
    bert_args = trainer_args(
        output_dir="./checkpoints",
        per_device_eval_batch_size=16,
        no_cuda=(args.device == "cpu"),
    )
    bert_trainer = training.SquadTrainer(
        model=bert_model, args=bert_args, data_collator=bert_dm.tokenizer,
    )

    bert_test_output = bert_trainer.predict(bert_dm.test_dataset)
    utils.save_answers(args.results, bert_test_output.predictions[-1])
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
        "-w", "--weights", help="path to the model checkpoint to load",
    )
    parser.add_argument(
        "-r",
        "--results",
        default=f"{os.getcwd()}/results.json",
        help="where to save computed predictions",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
