from dataclasses import dataclass


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
    neuron_health_death_threshold: float = -0.6
    # NH_birth
    neuron_health_birth_threshold: float = 0.2

    # DH_death
    dendrite_health_death_threshold: float = -0.6
    # DH_birth
    dendrite_health_birth_threshold: float = 0.2

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
