"""Compact two-layer fully connected LIF classifier."""

import snntorch as snn
import torch
from snntorch import surrogate
from torch import nn


class TwoLayerSNN(nn.Module):
    def __init__(self, n_inputs: int, n_hidden: int, n_classes: int, beta: float = 0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(n_hidden, n_classes)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, spike_train: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return output spike counts and per-layer spike activity."""
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        hidden_spikes, output_spikes = [], []

        for step_input in spike_train:
            current1 = self.fc1(step_input)
            spike1, mem1 = self.lif1(current1, mem1)
            current2 = self.fc2(spike1)
            spike2, mem2 = self.lif2(current2, mem2)
            hidden_spikes.append(spike1)
            output_spikes.append(spike2)

        hidden_activity = torch.stack(hidden_spikes)
        output_activity = torch.stack(output_spikes)
        return output_activity.sum(dim=0), {
            "hidden_spikes": hidden_activity,
            "output_spikes": output_activity,
        }
