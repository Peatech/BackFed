python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=multi-adversary \
    atk_config.num_adversaries_per_round=4 \
    checkpoint=2000 \
    num_rounds=200 \
    num_gpus=0.5 \
    num_cpus=1 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"5,4,3,2,1,0\"

python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=multi-adversary \
    atk_config.num_adversaries_per_round=4 \
    checkpoint=2000 \
    num_rounds=200 \
    num_gpus=0.5 \
    num_cpus=1 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"5,4,3,2,1,0\"