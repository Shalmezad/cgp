from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Config:

    num_inputs: list[int]
    num_outputs: list[int]

    # NN_max
    max_num_neurons: int = 30

    # N_init
    initial_non_output_neurons: int = 6

    # DN_max
    max_num_dendrites: int = 60

    # ND_init
    initial_num_dendrites: int = 5

    # NH_death
    neuron_health_death_threshold_pre: float = -0.6
    neuron_health_death_threshold_while: float = -0.58
    # NH_birth
    neuron_health_birth_threshold_pre: float = 0.308
    neuron_health_birth_threshold_while: float = 0.8

    # DH_death
    dendrite_health_death_threshold_pre: float = -0.404772
    dendrite_health_death_threshold_while: float = -0.38
    # DH_birth
    dendrite_health_birth_threshold_pre: float = -0.2012
    dendrite_health_birth_threshold_while: float = 0.85

    # \delta_sh
    soma_health_increment_pre: float = 0.1
    soma_health_increment_while: float = 0.01

    # \delta_sp
    soma_position_increment_pre: float = 0.1
    soma_position_increment_while: float = 0.01

    # \delta_sb
    soma_bias_increment_pre: float = 0.07
    soma_bias_increment_while: float = 0.0402

    # \delta_dh
    dendrite_health_increment_pre: float = 0.1
    dendrite_health_increment_while: float = 0.01

    # \delta_dp
    dendrite_position_increment_pre: float = 0.2032
    dendrite_position_increment_while: float = 0.01

    # \delta_dw
    dendrite_weight_increment_pre: float = 0.1
    dendrite_weight_increment_while: float = 0.02029

    # NDS_pre
    num_steps_pre_epoch: int = 8
    # NDS_while
    num_steps_during_epoch: int = 3

    # N_ep
    num_epochs: int = 8

    # Increment option:
    class NeuralValueIncrement(Enum):
        RAW_VALUE = 'RAW_VALUE'
        SIGMOID = 'SIGMOID'

    increment_option: NeuralValueIncrement = NeuralValueIncrement.SIGMOID
