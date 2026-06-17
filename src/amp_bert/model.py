"""Model construction and training-argument helpers."""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification, TrainingArguments

MODEL_NAME = "Rostlab/prot_bert_bfd"


def build_model(model_name: str = MODEL_NAME):
    """Fresh ProtBERT-BFD with a 2-class sequence-classification head."""
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


def model_init():
    """Zero-argument factory for ``Trainer(model_init=...)``.

    Must take no arguments: HuggingFace ``Trainer`` inspects the signature and,
    if it sees one parameter, calls it as ``model_init(trial)`` (passing
    ``None`` outside a hyper-parameter search), which would be misread as the
    model name. Every call starts from identical pre-trained weights.
    """
    return build_model(MODEL_NAME)


def build_training_args(
    output_dir: str,
    *,
    num_train_epochs: int = 15,
    learning_rate: float = 5e-5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 64,
    weight_decay: float = 0.1,
    logging_dir: str | None = None,
    fp16: bool = True,
    seed: int = 0,
    **overrides,
) -> TrainingArguments:
    """Recreate the paper's fine-tuning configuration with sane defaults.

    Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
    (= 64 here, matching the original notebook). Override any field via kwargs.
    """
    args = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=0,
        weight_decay=weight_decay,
        logging_dir=logging_dir or f"{output_dir}/logs",
        logging_steps=100,
        eval_strategy="no",
        save_strategy="no",
        fp16=fp16,
        run_name="AMP-BERT",
        seed=seed,
        report_to="none",
    )
    args.update(overrides)
    return TrainingArguments(**args)
