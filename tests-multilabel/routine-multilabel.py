import os 

from datasets import load_from_disk

from src.functions import clean_memory
from src.classification_pipe import ClassificationPipe
from src.prediciton_pipe import PredictionPipe

MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 512
N_CLASSES = 4
CLASSES = [
    'loneliness', 
    'emptiness', 
    'hopelessness', 
    'worthlessness', 
    'sadness', 
    'suicide intent', 
    'brain dysfunction (forget)', 
    'anger'
]
DEVICE = "cuda"
DEVICE_BATCH_SIZE = 8
TEXT_COLUMN = "text-clean-no-emoji"


pipe, dsd = None, None
for n_classes in range(2, len(CLASSES)): 
    output_dir = f"./models/multilabel-top-{n_classes}"

    dsd = load_from_disk("./data/multi-label-split/")
    pipe = ClassificationPipe(
        mode = "multilabel",
        text_column = TEXT_COLUMN,
        label_columns = CLASSES[:n_classes],
        output_dir = output_dir,
        model_name = MODEL_NAME, 
        device_batch_size = DEVICE_BATCH_SIZE, 
        device = DEVICE, 
        tokenizer_max_length = MAX_LENGTH,
        additional_training_arguments={
            "num_train_epochs" : 5,
            "gradient_accumulation_steps" : 2,
            "fp16" : True,
        }
    )
    print(pipe)
    dsd = pipe.tokenize(dsd, test_mode = True)

    pipe.train(dsd)
    config = pipe.save_important_info(f"./model-configs/multilabel-top-{n_classes}.json")

    for checkpoint in os.listdir(output_dir):
        pipe_pred = PredictionPipe(
            checkpoint_dir=f"{output_dir}/{checkpoint}",
            model_name = MODEL_NAME,
            text_column = TEXT_COLUMN, 
            labels = CLASSES[:n_classes],
            tokenizer_parameters = config["tokenizer_parameters"],
            device_batch_size = DEVICE_BATCH_SIZE, 
            device = DEVICE
        )
        predictions = pipe_pred.predict(dsd)
        for split in predictions:
            predictions[split].to_csv(f"./outputs/multilabel-top-{n_classes}-{split}.csv")
        os.system(f"rm -rf {output_dir}/{checkpoint}")


del pipe, dsd
clean_memory()