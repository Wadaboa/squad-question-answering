import collections
import time
import os
import importlib.util

import numpy as np
import sklearn
import torch
import transformers
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
    speed_metrics,
)
from transformers.integrations import WandbCallback
from transformers.file_utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available

import utils


class SquadTrainer(transformers.Trainer):
    """
    Custom HuggingFace Trainer class
    """

    def __init__(self, *args, **kwargs):
        kwargs["compute_metrics"] = self.compute_metrics
        super(SquadTrainer, self).__init__(*args, **kwargs)
        self.wandb_callback = None
        if is_wandb_available():
            self.wandb_callback = CustomWandbCallback()
            self.remove_callback(WandbCallback)
            self.add_callback(self.wandb_callback)

    def compute_loss(self, model, inputs):
        """
        Override loss computation to calculate and log metrics
        during training
        """
        outputs = model(**inputs)

        # Custom logging steps (to log training metrics)
        if (self.state.global_step == 1 and self.args.logging_first_step) or (
            self.args.logging_steps > 0
            and self.state.global_step > 0
            and self.state.global_step % self.args.logging_steps == 0
        ):
            labels = None
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            if has_labels:
                labels = nested_detach(
                    tuple(inputs.get(name) for name in self.label_names)
                )
                if len(labels) == 1:
                    labels = labels[0]

            # Compute and log metrics only if labels are available
            if labels is not None:
                metrics = self.compute_scores(
                    EvalPrediction(
                        predictions=(outputs["word_outputs"], outputs["indexes"]),
                        label_ids=labels,
                    )
                )
                if self.wandb_callback is not None:
                    self.wandb_callback.update_metrics(metrics)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        return outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    def compute_scores(self, eval_prediction):
        # Predictions should be a tuple containing
        # (word_outputs, indexes)
        preds = eval_prediction.predictions
        assert isinstance(preds, tuple)
        word_outputs, indexes = preds
        
        # Transfer to CPU
        if isinstance(word_outputs, torch.Tensor):
            word_outputs = word_outputs.cpu()
        if isinstance(indexes, torch.Tensor):
            indexes = indexes.cpu()

        if self.args.do_predict:
            df = self.test_dataset.df
        elif self.model.training:
            df = self.train_dataset.df
        else:
            df = self.eval_dataset.df

        start = df["answer_start"].iloc[indexes].tolist()
        end = df["answer_end"].iloc[indexes].tolist()
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(list(zip(t[0], t[1]))) for t in zip(start, end)],
            batch_first=True,
            padding_value=-100,
        )
        labels = utils.get_nearest_answers(labels, word_outputs, device="cpu")

        preds_dict = utils.from_words_to_text(
            df, word_outputs.tolist(), indexes.tolist(),
        )
        labels_dict = utils.from_words_to_text(df, labels.tolist(), indexes.tolist())
        return self.get_raw_scores(preds_dict, labels_dict)

    def compute_metrics(self, eval_prediction):
        """
        Custom function that computes task-specific
        training and evaluation metrics
        """
        scores = self.compute_scores(eval_prediction)
        return {k: np.mean(v) for k, v in scores.items()}

    def exact_match(self, label, pred):
        """
        Compute the EM score of the SQuAD competition,
        measuring the number of exactly matched answers
        """
        return float(label == pred)

    def accuracy_precision_recall(self, label, pred):
        label_tokens = label.split()
        pred_tokens = pred.split()
        common = collections.Counter(label_tokens) & collections.Counter(pred_tokens)
        num_same = float(sum(common.values()))

        # If either is no-answer, then 1 if they agree, 0 otherwise
        if len(label_tokens) == 0 or len(pred_tokens) == 0:
            res = float(label_tokens == pred_tokens)
            return res, res, res
        if num_same == 0:
            return 0.0, 0.0, 0.0

        num_preds, num_labels = len(pred_tokens), len(label_tokens)
        accuracy = num_same / (num_preds + num_labels - num_same)
        precision = num_same / num_preds
        recall = num_same / num_labels
        return accuracy, precision, recall

    def f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0
        return (2 * precision * recall) / (precision + recall)

    def get_raw_scores(self, preds_dict, labels_dict):
        scores = collections.defaultdict(list)
        for question_id in labels_dict.keys():
            pred = preds_dict[question_id]
            label = labels_dict[question_id]
            accuracy, precision, recall = self.accuracy_precision_recall(label, pred)
            f1 = self.f1_score(precision, recall)
            em = self.exact_match(label, pred)
            scores["accuracy"].append(accuracy)
            scores["precision"].append(precision)
            scores["recall"].append(recall)
            scores["f1"].append(f1)
            scores["em"].append(em)
        return scores

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        """
        Run prediction and returns predictions and potential metrics
        """
        if test_dataset is not None and not isinstance(
            test_dataset, collections.abc.Sized
        ):
            raise ValueError("test_dataset must implement __len__")

        # Test the model with the given dataloader and gather outputs
        self.test_dataset = test_dataset
        self.args.do_predict = True
        start_time = time.time()
        output = self.prediction_loop(
            self.get_test_dataloader(self.test_dataset),
            description="Test",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.args.do_predict = False

        # Compute answers (taking spans from original contexts)
        answers_dict = utils.from_words_to_text(
            test_dataset.df,
            output.predictions[-2].tolist(),
            output.predictions[-1].tolist(),
        )

        # Update metrics and patch the predictions attribute
        # with the computed answers
        output.metrics.update(
            speed_metrics(metric_key_prefix, start_time, len(test_dataset))
        )
        if self.wandb_callback is not None:
            self.wandb_callback.save_notes(output.metrics)
        return PredictionOutput(
            predictions=output.predictions + (answers_dict,),
            label_ids=output.label_ids,
            metrics=output.metrics,
        )


def is_wandb_available():
    if os.getenv("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES:
        return False
    return importlib.util.find_spec("wandb") is not None


class CustomWandbCallback(WandbCallback):
    def __init__(self):
        super().__init__()
        self.metrics = collections.defaultdict(list)

    def setup(self, args, state, model, reinit, **kwargs):

        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                run_name = trial_name
                init_args["group"] = args.run_name
            else:
                run_name = args.run_name

            self._wandb.init(
                project=os.getenv("WANDB_PROJECT", "squad-qa"),
                group=os.getenv("WANDB_RUN_GROUP", None),
                config=combined_dict,
                name=run_name,
                reinit=True,
                **init_args,
            )

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model,
                    log=os.getenv("WANDB_WATCH", "gradients"),
                    log_freq=max(100, args.logging_steps),
                )

            # Log outputs
            self._log_model = os.getenv(
                "WANDB_LOG_MODEL", "FALSE"
            ).upper() in ENV_VARS_TRUE_VALUES.union({"TRUE"})

    def update_metrics(self, metrics):
        for k, v in metrics.items():
            self.metrics[k].extend(v)

    def save_notes(self, notes):
        if self._wandb is not None and self._wandb.run is not None:
            self._wandb.run.notes = self._wandb.run.notes + "\n" + str(notes)
            self._wandb.run.save()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.metrics = collections.defaultdict(list)
        self._save_checkpoint(args.output_dir, state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        logs = {k: np.mean(v) for k, v in self.metrics.items()}
        if state.epoch is not None:
            logs["epoch"] = round(state.epoch, 2)
        self.on_log(args, state, control, logs=logs, **kwargs)
        output = {**logs, **{"step": state.global_step}}
        state.log_history.append(output)

    def on_train_end(self, args, state, control, **kwargs):
        self._save_checkpoint(args.output_dir, state.global_step)

    def _save_checkpoint(self, output_dir, step):
        checkpoint_path = f"{output_dir}/checkpoint-{step}"
        if os.path.exists(checkpoint_path):
            self._wandb.save(f"{checkpoint_path}/*")
