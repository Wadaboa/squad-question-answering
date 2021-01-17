import collections

import numpy as np
import sklearn
import transformers
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction

import utils


class SquadTrainer(transformers.Trainer):
    """
    Custom HuggingFace Trainer class
    """

    def __init__(self, *args, **kwargs):
        super(SquadTrainer, self).__init__(*args, **kwargs)

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
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=outputs["outputs"], label_ids=labels)
                )
                self.log(metrics)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        return outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    
    def predict(
        self, test_dataset, ignore_keys= None, metric_key_prefix= "test"
    ) :
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        .. note::

            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        if test_dataset is not None and not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")

        test_dataloader = self.get_test_dataloader(test_dataset)
        
        start_time = time.time()
        output = self.prediction_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        
        
        
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))
        return output

    
    
def exact_match(labels, preds):
    """
    Compute the EM score of the SQuAD competition,
    measuring the number of exactly matched answers
    """
    assert labels.shape == preds.shape
    total = labels.shape[0]
    matches = np.count_nonzero((labels == preds).all(axis=1))
    return matches / total


def compute_metrics(eval_prediction):
    """
    Custom function that computes task-specific
    training and evaluation metrics
    """
    # Labels are stored as a single tensor
    # (concatenation of answer start and answer end)
    labels = eval_prediction.label_ids
    preds = eval_prediction.predictions
    labels = utils.get_nearest_answers(labels, preds)
    
    labels, preds = labels.numpy(), preds.numpy()
    f_labels, f_preds = labels.flatten(), preds.flatten()

    # Return a dictionary of metrics, as required by the Trainer
    return {
        "f1": sklearn.metrics.f1_score(f_labels, f_preds, average="macro"),
        "accuracy": sklearn.metrics.accuracy_score(f_labels, f_preds),
        "em": exact_match(labels, preds),
    }
