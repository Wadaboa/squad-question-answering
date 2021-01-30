import argparse
import os

import numpy as np
import torch

import dataset
import tokenizer
import model
import training
import utils
import config


RECURRENT_MODELS = ("baseline", "bidaf")
TRANSFORMER_MODELS = ("bert", "distilbert", "electra")
MODELS = {
    "baseline": model.QABaselineModel,
    "bidaf": model.QABiDAFModel,
    "bert": model.QABertModel,
    "distilbert": model.QADistilBertModel,
    "electra": model.QAElectraModel,
}


def load_recurrent_model(model_type, device):
    """
    Return the specified PyTorch recurrent-based model
    and the corresponding tokenizer
    """
    embedding_model, vocab = utils.load_embedding_model(
        config.EMBEDDING_MODEL_NAME,
        embedding_dimension=config.EMBEDDING_DIMENSION,
        unk_token=config.UNK_TOKEN,
        pad_token=config.PAD_TOKEN,
    )
    embedding_layer = layer_utils.get_embedding_module(
        embedding_model, pad_id=vocab[PAD_TOKEN]
    )

    tokenizer = tokenizer.get_recurrent_tokenizer(
        vocab,
        config.MAX_CONTEXT_TOKENS,
        config.UNK_TOKEN,
        config.PAD_TOKEN,
        device=device,
    )
    model = MODELS[model_type](embedding_layer, device=device)
    return tokenizer, model


def load_transformer_model(model_type, device):
    """
    Return the specified PyTorch transformer-based model
    and the corresponding tokenizer
    """
    transformer_tokenizer = tokenizer.get_transformer_tokenizer(
        config.BERT_VOCAB_PATH, config.MAX_BERT_TOKENS, device=device
    )
    model = MODELS[model_type](device=device)
    return tokenizer, model


def load_tokenizer_and_model(model_type, device):
    """
    Return the specified PyTorch model and tokenizer
    """
    if model_type in RECURRENT_MODELS:
        return load_recurrent_model(model_type, device)
    elif model_type in TRANSFORMER_MODELS:
        return load_transformer_model(model_type, device)
    else:
        raise ValueError("Unsupported model")


def main(args):
    """
    Predict answers to the given test questions
    """
    squad_dataset = dataset.SquadDataset(test_set_path=args.path)
    squad_tokenizer, squad_model = load_tokenizer_and_model(args.model, args.device)
    data_manager = dataset.SquadDataManager(
        squad_dataset, squad_tokenizer, device=args.device
    )

    squad_model.load_state_dict(torch.load(args.weights, map_location=args.device))
    print(f"Weights loaded from {args.weights}")

    trainer_args = utils.get_default_trainer_args()(
        output_dir="./checkpoints",
        per_device_eval_batch_size=args.batch,
        no_cuda=(args.device == "cpu"),
    )
    trainer = training.SquadTrainer(
        model=squad_model, args=trainer_args, data_collator=data_manager.tokenizer,
    )

    test_output = trainer.predict(data_manager.test_dataset)
    utils.save_answers(args.results, test_output.predictions[-1])
    print(f"Results saved in {args.results}")


def parse_args():
    """
    Parse command-line arguments
    """
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
        default="electra",
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
