class Config:

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
