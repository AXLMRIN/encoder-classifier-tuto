from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch import Tensor
from torch.nn import Sigmoid
from transformers import EvalPrediction
import numpy as np

def multi_label_metrics(results_matrix, labels : Tensor, threshold : float = 0.5
                        ) -> dict:
    '''Taking a results matrix (batch_size x num_labels), the function (with a 
    threshold) associates labels to the results => y_pred
    From this y_pred matrix, evaluate the f1_micro, roc_auc and accuracy metrics
    '''
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = Sigmoid()
    probs = sigmoid(Tensor(results_matrix))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    return {'f1_micro': f1_micro_average,
            'f1_macro': f1_macro_average,
             'roc_auc': roc_auc,
             'accuracy': accuracy}

def compute_metrics(model_output):
    if isinstance(model_output.predictions,tuple):
        results_matrix = model_output.predictions[0]
    else:
        results_matrix = model_output.predictions

    metrics = multi_label_metrics(results_matrix=results_matrix, 
        labels=model_output.label_ids)
    return metrics

# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ===

from datasets import DatasetDict, Dataset
import numpy as np 
import pandas as pd 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import Tensor

RANDOM_SEED = 2306406
DATA_PATH = "./data/multilabel_classification_biotech_event.csv"
MODEL_NAME = "google-bert/bert-base-uncased"
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
DEVICE = "cpu"

# Open data
df = pd.read_csv(DATA_PATH)

# Create splits 
N = len(df)
N_train = int(N * 0.7)
N_train_eval = int(N * 0.1)
N_test = int(N * 0.1)
N_final_eval = N - N_train - N_train_eval - N_test 

assert N_final_eval > 0


indices = df.index.to_series()
indices_train = (
    indices
    .sample(n = N_train, random_state=RANDOM_SEED)
)
indices_train_eval = (
    indices
    .drop(index=indices_train.index)
    .sample(n = N_train_eval, random_state=RANDOM_SEED)
)
indices_test = (
    indices
    .drop(index=[*indices_train.index, *indices_train_eval.index])
    .sample(n = N_train_eval, random_state=RANDOM_SEED)
)
indices_final_test = (
    indices
    .drop(index = [*indices_train, *indices_train_eval,*indices_test])
)

df_split = (
    pd.concat({
        "train"         : df.loc[indices_train      , :],
        "train_eval"    : df.loc[indices_train_eval , :],
        "test"          : df.loc[indices_test       , :],
        "final_test"     : df.loc[indices_final_test  , :],
    })
    .reset_index()
    .drop(columns=["level_1"])
    .rename(columns = {"level_0": "split"})
)

# Load model
labels = CLASSES[:N_CLASSES]
num_labels = N_CLASSES
id2label = {id:label for id, label in enumerate(labels)}
label2id = {label:id for id, label in enumerate(labels)}


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,               
    problem_type = "multi_label_classification"                   
).to(device=DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create a dataset from the splits we created before
grouped_ds_split = df_split.groupby("split")
dsd = DatasetDict({
    split :( 
        Dataset
        .from_pandas(grouped_ds_split.get_group(split))
        .with_format("torch", device=DEVICE)#, dtype=int)
    )
    for split in ["train", "train_eval", "test", "final_test"]
})

tokenizer_parameters = {
    "truncation":True, 
    "padding":"max_length",
    "max_length":400,
    "return_tensors":"pt"
}

def preprocess_dataset(row: dict):
    tokenized_entry = tokenizer(row["text-clean-no-emoji"], **tokenizer_parameters)
    id_label_as_tensor = Tensor([
        int(row[label]) for label in labels
    ])
    return {
        **row.copy(),
        "labels": id_label_as_tensor,
        "attention_mask" : tokenized_entry["attention_mask"].reshape(-1),
        "input_ids" : tokenized_entry["input_ids"].reshape(-1)
    }


dsd = dsd.map(preprocess_dataset, batch_size=32)

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding


training_arguments = TrainingArguments(
    # Hyperparameters
    # num_train_epochs = 5,
    num_train_epochs = 7,
    learning_rate = 5e-5,
    weight_decay  = 0.0,
    warmup_ratio  = 0.0,
    optim = "adamw_torch_fused",
    # Second order hyperparameters
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps = 8,
    # Pipe
    output_dir = "./models/training",
    overwrite_output_dir=True,
    
    logging_strategy = "epoch",
    # eval_strategy = "epoch",
    eval_strategy = "steps",
    eval_steps = 32,
    save_strategy = "epoch",
    # load_best_model_at_end = True,
    # save_total_limit = 5 + 1,

    disable_tqdm = False,
)

trainer = Trainer(
    model = model, 
    args = training_arguments,
    train_dataset=dsd["train"],
    eval_dataset=dsd["train_eval"],
    compute_metrics = compute_metrics
)

trainer.train()
# WARNING MODEL switches to mps need to handle that