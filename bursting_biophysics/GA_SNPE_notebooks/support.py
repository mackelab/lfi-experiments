from copy import deepcopy
import torch
from torch import Tensor
from tqdm.auto import tqdm
from warnings import warn


class PriorRejectionProposal:
    def __init__(
        self,
        posterior,
        prior,
        num_samples_to_estimate: int = 10_000,
        quantile: float = 0.0,
        log_prob_offset: float = 0.0,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self._dim_theta = int(prior.sample((1,)).shape[1])
        self._posterior = deepcopy(posterior)
        self._posterior.net.eval()
        self._prior = prior
        self._thr = identify_cutoff(
            posterior, num_samples_to_estimate, quantile, log_prob_offset
        )

        if device == "cpu":
            warn(
                "You are using `cpu` to sample from the `PriorRejectionProposal`. "
                "This will be much slower than sampling with `cuda`."
            )
        self._device = device

        self._posterior.net = self._posterior.net.to(self._device)
        self._posterior = self._posterior.set_default_x(
            self._posterior.default_x.to(self._device)
        )
        self._posterior.net._distribution = self._posterior.net._distribution.to(self._device)
        self._posterior.net._transform = self._posterior.net._transform.to(self._device)
        self._posterior._device = self._device

    def sample(
        self,
        sample_shape,
        show_progress_bars: bool = False,
        max_sampling_batch_size: int = 10_000,
    ) -> Tensor:

        num_samples = torch.Size(sample_shape).numel()
        num_sampled_total, num_remaining = 0, num_samples
        accepted, acceptance_rate = [], float("Nan")

        # Progress bar can be skipped.
        pbar = tqdm(
            disable=not show_progress_bars,
            total=num_samples,
            desc=f"Drawing {num_samples} posterior samples",
        )

        # To cover cases with few samples without leakage:
        sampling_batch_size = min(num_samples, max_sampling_batch_size)
        while num_remaining > 0:
            print("Round")
            # Sample and reject.
            candidates = self._prior.sample((sampling_batch_size,)).reshape(
                sampling_batch_size, -1
            )
            candidates = candidates.to(self._device)
            are_accepted_by_classifier = self.log_prob(candidates)
            samples = candidates[are_accepted_by_classifier.bool()]
            accepted.append(samples)

            # Update.
            num_sampled_total += sampling_batch_size
            num_remaining -= samples.shape[0]
            pbar.update(samples.shape[0])

            # To avoid endless sampling when leakage is high, we raise a warning if the
            # acceptance rate is too low after the first 1_000 samples.
            acceptance_rate = (num_samples - num_remaining) / num_sampled_total
            print("acceptance_rate", acceptance_rate)

            # For remaining iterations (leakage or many samples) continue sampling with
            # fixed batch size.
            sampling_batch_size = max_sampling_batch_size

        pbar.close()
        print(
            f"The classifier rejected {(1.0 - acceptance_rate) * 100:.1f}% of all "
            f"samples. You will get a speed-up of "
            f"{(1.0 / acceptance_rate - 1.0) * 100:.1f}%.",
        )

        # When in case of leakage a batch size was used there could be too many samples.
        samples = torch.cat(accepted)[:num_samples]
        assert (
            samples.shape[0] == num_samples
        ), "Number of accepted samples must match required samples."

        return samples

    def log_prob(self, theta: Tensor) -> Tensor:
        log_probs = self._posterior.log_prob(theta)
        predictions = log_probs > self._thr
        return predictions.float()
    
def identify_cutoff(
    posterior,
    num_samples_to_estimate: int = 10_000,
    quantile: float = 0.0,
    log_prob_offset: float = 0.0,
) -> Tensor:
    posterior.net.eval()
    samples = posterior.sample((num_samples_to_estimate,))
    sample_probs = posterior.log_prob(samples)
    sorted_probs, _ = torch.sort(sample_probs)
    return sorted_probs[int(quantile * num_samples_to_estimate)] + log_prob_offset
