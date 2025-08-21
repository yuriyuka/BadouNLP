Config = {
    "pretrain_model_path": r"/bert-base-chinese",

    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "max_length": 32,
    "class_num": 9,
    "hidden_size": 256,
    "seed": 10086,
    "tuning_tactics": "lora_tuning",
    "use_crf":False,

    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "model_path": "model_output",
    "vocab_path": "chars.txt",
    "schema_path": "ner_data/schema.json",

    "model_type": "bert"
}
