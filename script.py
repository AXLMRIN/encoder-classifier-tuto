from typing import Any
from gc import collect as gc_collect

from datasets import Dataset
import matplotlib.pyplot as plt
import pandas as pd 
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from torch import Tensor, Size
from torch.cuda import is_available as cuda_available
from torch.cuda import empty_cache, synchronize, ipc_collect

# CONFIG
SEED = 2306406
MODEL_NAME = "distilbert/distilbert-base-uncased"
# MODEL_NAME = "google-bert/bert-base-uncased"

DEVICE = "cuda" if cuda_available() else "cpu"
# CODE ==================================================
print("# I- Open Data")
df = pd.read_csv("./data/ibc.csv")
print(df.head())

df["sentence-len"] = df["sentence"].apply(len)
plt.figure()
ax = df["sentence-len"].hist()
plt.savefig('./outputs/sentence-len-hist.png')
plt.close()
plt.figure()
ax = df.hist("sentence-len", by="leaning"); 
plt.savefig('./outputs/sentence-len-hist-by-label.png')
plt.close()

print("\t i- Preprocess Dataset")
df = df.loc[df["sentence-len"] > 50,:]
df["ID"] = [f"ID-{i:04}" for i in range(len(df))]
df = df.set_index("ID")

labels = list(df["leaning"].unique())
num_labels = len(labels)
id2label = {id:label for id, label in enumerate(labels)}
label2id = {label:id for id, label in enumerate(labels)}

print(f'''
num_labels : {num_labels}
id2label : {id2label}
label2id : {label2id}
'''
)


print("\t ii- Preprocess texts")
def preprocess_text(text: str):
    if not(isinstance(text, str)):
        return pd.NA
    return (
        text
        .replace("``", '"')
        .replace("''", '"')
        .replace(" ,", ",")
        .replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
        .replace(" :", ":")
        .replace(" 's", "'s")
    )
df["sentence-preprocessed"] = df["sentence"].apply(preprocess_text)

ds = Dataset.from_pandas(df[["sentence-preprocessed","leaning"]])
ds = ds.with_format("torch")
ds = ds.rename_columns({
    "sentence-preprocessed":"text",
    "leaning":"labels_text"
})

print("\t iii- Split into train, test, eval datasets")
dsd = ds.train_test_split(test_size = 0.2, shuffle=True, seed=SEED)

# Create an eval dataset
temp = dsd["train"].train_test_split()
dsd["train"] = temp["train"]
dsd["eval"] = temp["test"]
print(dsd)

print("# II- Prepare Training")
def clean_memory():
    empty_cache()
    if cuda_available(): synchronize();ipc_collect()
    gc_collect()
    print("Memory clean")

try:
    # Instanciate variables
    trainer, classif_model, tokenizer, input_ids_ex, attention_mask_ex = (None,)*5
    
    print("\t i- Load models")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    classif_model = (
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id                                            
        )
        .to(device=DEVICE)
    )
    print(classif_model)
    
    print("\t ii- Display example")
    entry = [
        "Hello World",
        "This is a second query"
    ]
    
    tokenizer_parameters = {
        "truncation":True, 
        "padding":"max_length",
        "max_length":400,
        "return_tensors":"pt"
    }

    model_input = tokenizer(entry,**tokenizer_parameters)
    input_ids_ex = model_input["input_ids"].to(device=DEVICE)
    attention_mask_ex = model_input["attention_mask"].to(device=DEVICE)
    print("model device: ", classif_model.device)
    print("model input['input_ids'] device: ", input_ids_ex.device)
    print("model input['attention_mask'] device: ", attention_mask_ex.device)
    
    base_model_output = classif_model.base_model(
        input_ids = input_ids_ex,
        attention_mask = input_ids_ex
    )
    classif_model_output = classif_model(
        input_ids = input_ids_ex,
        attention_mask = input_ids_ex
    )
    del input_ids_ex, attention_mask_ex # Delete elements from cuda to free space
    input_ids_ex, attention_mask_ex = None, None # Reinitiate the values to prevent raising errors in the finally section
    
    print(f'''
    # model input keys: {', '.join(model_input)}
    model input shape (pytorch tensor): {model_input["input_ids"].shape}
    base model output keys: {', '.join(base_model_output)}
    base model output last_hidden_state shape (pytorch tensor): {base_model_output.last_hidden_state.shape}
    classification model output key: {', '.join(classif_model_output)}
    classification model output logits shape (pytorch tensor): {classif_model_output.logits.shape}
    ''')
    
    print("\t iii- Tokenize dataset")
    def preprocess_dataset(row: dict[str:Any]):
        tokenized_entry = tokenizer(row["text"], **tokenizer_parameters)
        id_label = [int(label2id[row["labels_text"]])]
        id_label_as_tensor = Tensor([int(i == id_label) for i in range(num_labels)])
        return {
            **row.copy(),
            "labels": id_label_as_tensor.to(device=DEVICE),
            "attention_mask" : tokenized_entry["attention_mask"].reshape(-1).to(device=DEVICE),
            "input_ids" : tokenized_entry["input_ids"].reshape(-1).to(device=DEVICE)
        }
    
    dsd = dsd.map(preprocess_dataset, batch_size=32)
    
    print("\t iv- Create Trainer")
    training_arguments = TrainingArguments(
        # Hyperparameters
        num_train_epochs = 5,
        learning_rate = 5e-5,
        weight_decay  = 0.0,
        warmup_ratio  = 0.0,
        optim = "adamw_torch_fused",
        # Second order hyperparameters
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps = 8,
        # Metrics
        # metric_for_best_model="f1_macro",
        # Pipe
        output_dir = "./models/training",
        overwrite_output_dir=True,
        eval_strategy = "epoch",
        logging_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        save_total_limit = 5 + 1,
    
        disable_tqdm = False,
    )
    
    trainer = Trainer(
        model = classif_model, 
        args = training_arguments,
        train_dataset=dsd["train"].select(range(0,1000)),
        eval_dataset=dsd["eval"].select(range(0,500)),
    )
    
    print("\t v- Train")
    trainer.train()

except Exception as e:
    print(e)

finally:
    del trainer, classif_model, dsd, ds, df, tokenizer, input_ids_ex, attention_mask_ex
    clean_memory()