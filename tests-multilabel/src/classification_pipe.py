import json
from mergedeep import merge as dict_merge

from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from torch import Tensor

from .functions import compute_metrics, clean_memory

class ClassificationPipe:
    def __init__(
        self, 
        mode: str,
        text_column : str, 
        label_columns : list[str], 
        output_dir: str, 
        model_name : str, 
        device_batch_size : int,
        unique_labels : list[str] | None = None, 
        device : str | None = None, 
        tokenizer_max_length : int | None = None, 
        additional_training_arguments : TrainingArguments|None = None
        ) -> None :
        """
        Pipeline for Multiclass or Multilabel
        """
        assert mode in ["multiclass", "multilabel"]
        self.__mode = mode
        self.__device = device if device is not None else "cpu"
        self.__text_column = text_column

        if self.__mode == "multiclass":
            assert isinstance(label_columns, list) and len(label_columns) == 1
            assert unique_labels is not None

            self.__label_column = label_columns[0]
            self.__labels = list(unique_labels)
        elif self.__mode == "multilabel": 
            assert isinstance(label_columns, list)
            self.__labels = list(label_columns)
            self.__label_column = None

        self.__num_labels = len(self.__labels)
        self.__id2label = {id:label for id, label in enumerate(self.__labels)}
        self.__label2id = {label:id for id, label in enumerate(self.__labels)}

        self.__model_name = model_name
        self.__tokenizer = AutoTokenizer.from_pretrained(self.__model_name)
        self.__model = AutoModelForSequenceClassification.from_pretrained(
            self.__model_name, 
            num_labels=self.__num_labels,
            id2label=self.__id2label,
            label2id=self.__label2id,  
            problem_type = "multi_label_classification" if self.__mode == "multilabel" else None                        
        ).to(device=self.__device)
        
        self.__tokenizer_parameters = {
            "truncation":True, 
            "padding":"max_length",
            "max_length": tokenizer_max_length if tokenizer_max_length is not None else 512,
            "return_tensors":"pt"
        }

        self.__training_arguments_dict = {
            # Hyperparameters
            # num_train_epochs = 5,
            "num_train_epochs" : 7,
            "learning_rate" : 5e-5,
            "weight_decay" : 0.0,
            "warmup_ratio" : 0.0,
            "optim" : "adamw_torch_fused",
            # Second order hyperparameters
            "per_device_train_batch_size" : 4,
            "per_device_eval_batch_size" : 4,
            "gradient_accumulation_steps" : 8,
            # Pipe
            "output_dir" : "./models/training",
            "overwrite_output_dir" : True,
            
            "logging_strategy" : "epoch",
            # eval_strategy = "epoch",
            "eval_strategy" : "steps",
            "eval_steps" : 32,
            "save_strategy" : "epoch",
            # load_best_model_at_end = True,
            # save_total_limit = 5 + 1,

            "disable_tqdm" : False,
        }
        dict_merge(
            # Default training arguments
            self.__training_arguments_dict, 
            # Overwrite training arguments
            {
                "output_dir" : output_dir,
                "per_device_train_batch_size" : device_batch_size,
                "gradient_accumulation_steps" : 32 // device_batch_size,
                "per_device_eval_batch_size" : device_batch_size * 2,
                "use_cpu":  self.__device == "cpu",
                "use_mps_device":  self.__device == "mps",
                "no_cuda" : self.__device != "cuda",
                "dataloader_pin_memory" : self.__device == "cpu",
            },
            # include edditional training arguments
            additional_training_arguments if additional_training_arguments is not None else {}
        )
        self.__training_arguments = TrainingArguments(**self.__training_arguments_dict)

    def __str__(self):
        return "\n".join([
            f"Piepline training {self.__model_name} in {self.__mode}",
            f"Labels : {self.__labels}"
        ])

    def tokenize(self, dsd: DatasetDict): 

        def preprocess_dataset(row: dict):
            tokenized_entry = self.__tokenizer(row[self.__text_column], **self.__tokenizer_parameters)
            if self.__mode == "multiclass": 
                id_label = self.__label2id[row[self.__label_column]]
                id_label_as_tensor = [int(id_label == i) for i in range(self.__num_labels)]
                id_label_as_tensor = Tensor(id_label_as_tensor)
            elif self.__mode == "multilabel":
                id_label_as_tensor = [int(row[label]) for label in self.__labels]
                id_label_as_tensor = Tensor(id_label_as_tensor)
            return {
                **row.copy(),
                "labels": id_label_as_tensor,
                "attention_mask" : tokenized_entry["attention_mask"].reshape(-1),
                "input_ids" : tokenized_entry["input_ids"].reshape(-1)
            }

        dsd = dsd.map(preprocess_dataset, batch_size=32)
        
        for split in dsd:
            dsd[split] = dsd[split].with_format("torch", device=self.__device)

        return dsd
    
    def train(self, dsd : DatasetDict, test_mode : bool = False):
        if test_mode : 
            print("TEST - train : 50 - train_eval : 50")
            dsd["train"] = dsd["train"].select(range(50))
            dsd["train_eval"] = dsd["train_eval"].select(range(50))
            
        try:
            trainer = Trainer(
                model = self.__model, 
                args = self.__training_arguments,
                train_dataset=dsd["train"],
                eval_dataset=dsd["train_eval"],
                compute_metrics = compute_metrics
            )
            print("Training happening on : ", self.__training_arguments.device, " asked to train on ", self.__device," — ", self.__training_arguments.use_cpu, self.__training_arguments.use_mps_device, self.__training_arguments.no_cuda)

            trainer.train()
            # WARNING MODEL switches to mps need to handle that
        except Exception as e:
            print("# ERROR" + "#" * 93)
            print(e)
            print("#" * 100)

        finally:
            del self.__tokenizer, self.__model, dsd, trainer # All objects that are moved to the GPU
            clean_memory()

    def save_important_info(self, output_file : str):
        important_info = {
            "model_name" : self.__model_name,
            "mode" : self.__mode,
            "tokenizer_parameters": self.__tokenizer_parameters,
            "output_dir" : self.__training_arguments.output_dir,
            "device-requested" : self.__device,
            "device-used": str(self.__training_arguments.device), 
            "text_column" : self.__text_column,
            "label_column" : self.__label_column,
            "labels" : self.__labels,
            "num_labels" : self.__num_labels,
            "id2label" : self.__id2label,
            "label2id" : self.__label2id,
            "training_arguments" : self.__training_arguments_dict,
        }
        with open(output_file, "w") as file: 
            json.dump(important_info, file, indent = 4, ensure_ascii = True)
