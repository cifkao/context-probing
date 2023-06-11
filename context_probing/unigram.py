from typing import Any, Callable, Dict, List, Optional, Union
import logging

import torch
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutput


logger = logging.getLogger(__name__)


def estimate_unigram_logprobs(
    model: Callable[..., CausalLMOutput],
    tokenizer: PreTrainedTokenizerBase,
    max_steps: int = 20,
    convergence_threshold: float = 1e-4,
    initial_probs: Optional[torch.Tensor] = None,
    model_device: Optional[Union[torch.device, str]] = None,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Estimate the marginal log-probability of each token in the vocabulary.

    This is done by considering the Markov chain given by the bigram distribution and finding a
    stationary distribution using power iteration, starting from the distribution given by
    `initial_probs`.

    Args:
        model: A causal language model to use for estimating the probabilities.
        tokenizer: The tokenizer corresponding to the model.
        max_steps: The maximum number of steps to take when estimating the probabilities.
        convergence_threshold: The maximum change in log-probability for any token before
            considering the estimation to have converged.
        initial_probs: The initial distribution to use for the unigram probabilities. If None,
            a uniform distribution is used.
        model_device: The device to use for inputs to the model.
        batch_size: The batch size to use when computing the bigram probabilities.
    """
    if model_device is None:
        model_device = next(model.parameters()).device

    with torch.no_grad():
        bigram_probs = get_bigram_logits(
            model, tokenizer, model_device=model_device, batch_size=batch_size
        ).softmax(1)

        if initial_probs is None:
            unigram_probs = torch.ones(tokenizer.vocab_size, device=bigram_probs.device)
            unigram_probs /= tokenizer.vocab_size
        else:
            unigram_probs = initial_probs.detach().clone().to(bigram_probs.device)

        # Iteratively recompute the unigram probabilities until convergence
        for step in range(max_steps):
            new_unigram_probs = unigram_probs @ bigram_probs
            new_unigram_probs /= new_unigram_probs.sum()
            change = (new_unigram_probs.log() - unigram_probs.log()).nan_to_num().abs().max().item()
            logger.debug("Step %d: max(abs(diff(log(prob)))) = %f", step, change)
            if change < convergence_threshold:
                break
            unigram_probs = new_unigram_probs

        return unigram_probs.log()


def get_bigram_logits(
    model: Callable[..., CausalLMOutput],
    tokenizer: PreTrainedTokenizerBase,
    model_device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Get the logits for all bigrams (pairs of tokens from the vocabulary) for the given model.

    Returns a matrix of shape (vocab_size, vocab_size) where the entry at (i, j) is the logit for
    token j given token i as context (i.e. the bigram (i, j)).
    """
    logits = torch.empty(tokenizer.vocab_size, tokenizer.vocab_size)
    with torch.inference_mode():
        for i in range(0, tokenizer.vocab_size, batch_size):
            start, end = i, min(tokenizer.vocab_size, i + batch_size)
            logits[start:end] = (
                model(input_ids=torch.arange(start, end, device=model_device).unsqueeze(1))
                .logits.squeeze(1)
                .to(logits.device)
            )
    return logits
