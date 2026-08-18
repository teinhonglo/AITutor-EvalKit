"""Reusable inference runtime for the paper-best LoMTL evaluator."""

import logging
import threading
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .utils.constants import LABEL_LIST, SEED
from .utils.prompt import EvaluationDatasetFormatter

logger = logging.getLogger(__name__)

BASE_MODEL = "google/gemma-2-2b-it"
ADAPTER_PATH = Path(__file__).resolve().parents[2] / "assets/model/lora_model"
DIMENSIONS = (
    "Mistake_Identification",
    "Mistake_Location",
    "Providing_Guidance",
    "Actionability",
)
MAX_NEW_TOKENS = 10


class InvalidPredictionError(RuntimeError):
    """Raised when LoMTL generates text outside its fixed label set."""


def load_model_and_tokenizer(
    base_model_name=BASE_MODEL,
    adapter_path=ADAPTER_PATH,
    enable_lora=True,
    require_cuda=False,
):
    """Load a causal LM and optionally merge its LoRA adapter.

    This is shared by the original batch evaluator and the resident HTTP
    evaluator so both paths use the same model-loading behavior.
    """
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the LoMTL API, but no visible CUDA device was found. "
            "Check CUDA_VISIBLE_DEVICES and your PyTorch CUDA installation."
        )

    set_seed(SEED)
    logger.info("Loading tokenizer: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model: %s", base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    )
    if enable_lora and adapter_path:
        logger.info("Loading LoRA adapter: %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        try:
            model = model.merge_and_unload()
            logger.info("Merged LoRA adapter")
        except Exception:
            logger.exception("Could not merge LoRA adapter; using the attached adapter")
    else:
        logger.info("No LoRA adapter requested")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, tokenizer


class AutoEvaluator:
    """Resident, deterministic evaluator for the four paper-best LoMTL tasks."""

    def __init__(self):
        self.model, self.tokenizer = load_model_and_tokenizer(require_cuda=True)
        self.device = next(self.model.parameters()).device
        self._inference_lock = threading.Lock()

    def _evaluate_dimension(self, conversation_history, tutor_response, task):
        formatted = EvaluationDatasetFormatter.prepare_input(
            tokenizer=self.tokenizer,
            conv=conversation_history,
            resp=tutor_response,
            task=task,
            label_definitions=True,
            flag="eval",
        )
        inputs = self.tokenizer(formatted["text"], return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prediction = self.tokenizer.decode(
            outputs[0, input_length:], skip_special_tokens=True
        ).strip().split("\n", 1)[0].strip()
        if prediction not in LABEL_LIST:
            raise InvalidPredictionError(
                f"LoMTL generated invalid label {prediction!r} for {task}; "
                f"expected one of {LABEL_LIST}."
            )
        return prediction

    def evaluate(self, conversation_history: str, tutor_response: str) -> dict[str, str]:
        """Evaluate one tutor response without reloading model state."""
        with self._inference_lock:
            return {
                task: self._evaluate_dimension(conversation_history, tutor_response, task)
                for task in DIMENSIONS
            }
