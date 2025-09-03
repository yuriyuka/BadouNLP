Config = {
    "model_path": "/Library/workerspace/python_test/badou_demo1/week13/ner/output",
    "schema_path": "/Library/workerspace/python_test/badou_demo1/week13/ner/data/schema.json",
    "train_data_path": "/Library/workerspace/python_test/badou_demo1/week13/ner/data/train",
    "valid_data_path": "/Library/workerspace/python_test/badou_demo1/week13/ner/data/test",
    "vocab_path": "/Library/workerspace/python_test/badou_demo1/week13/ner/data/chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "bert_path": "/Library/workerspace/python_test/badou_demo1/week13/ner/bert-base-chinese"
}
