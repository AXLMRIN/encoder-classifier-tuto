RANDOM_SEED = 2306406

import pandas as pd 

url = "https://raw.githubusercontent.com/yiweiluo/GWStance/refs/heads/master/3_stance_detection/1_MTurk/full_annotations.tsv"
df_raw = pd.read_csv(url, sep = "\t") # Careful here, this document is a tsv (separator="\t" and not a csv (separator = ",")

df = df_raw.loc[:,["sent_id", "sentence", "MACE_pred", "av_rating"]]
df = df.rename(columns={"MACE_pred" : "label_text"}) # Rename MACE_pred for conveniency

print(df.groupby(["label_text"]).size())

df["sentence-len"] = df["sentence"].apply(len)
df.hist(column = "sentence-len", by="label_text")
print(df.groupby("label_text")["sentence-len"].describe())

df["sentence-len"].hist()
print(df["sentence-len"].describe())

def preprocess_text(text: str):
    if not(isinstance(text, str)):
        return pd.NA
    return (
        text
        .replace("’", "'")
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

print(df.groupby("sentence-preprocessed").size().value_counts())

df_no_duplicates = df.loc[~df["sent_id"].str.startswith("s"), :] 
print(df_no_duplicates.groupby("sentence-preprocessed").size().value_counts())

df_concensus = df_no_duplicates.groupby("sentence-preprocessed")["label_text"].agg(concensus = lambda X : len(set(X)) == 1)
print(df_concensus[df_concensus["concensus"] == False])

df_no_duplicates = df_no_duplicates.loc[
    df_no_duplicates["sentence-preprocessed"] != "We need to get rid of fossil fuel subsidies now.",
    :
]
df_no_duplicates = df_no_duplicates.drop_duplicates("sentence-preprocessed")

last_duplicates = [
    ("There is no solid evidence of global warming.","There is not solid evidence of global warming."),
    ("Balance of evidence suggests a discernible human influence on global climate.","The balance of evidence suggests a discernible human influence on global climate."),
    ("The alleged “ consensus ” behind the dangers of anthropogenic global warming is not nearly as settled among climate scientists as people imagine.","The alleged â consensus â behind the dangers of anthropogenic global warming is not nearly as settled among climate scientists as people imagine."),
    ("Rising global temperatures during the 19th and 20th centuries may be linked to greater plant photosynthesis.","Rising global temperatures during the 19th and 20th centuries could be linked to greater plant photosynthesis."),
    ("Climate change will continue to affect all types of weather phenomena and subsequently impact increasingly urbanised areas.","Climate change will continue to affect all types of weather phenomena and subsequently impact increasingly urbanized areas."),
]

for (s1, s2) in last_duplicates:
    lab_s1 = df_no_duplicates.loc[df_no_duplicates["sentence"] == s1, "label_text"]
    lab_s2 = df_no_duplicates.loc[df_no_duplicates["sentence"] == s2, "label_text"]
    if lab_s1.item() == lab_s2.item() : 
        df_no_duplicates.drop(index = lab_s2.index)
    else: 
        df_no_duplicates.drop(index = [*lab_s1.index, *lab_s2.index])

import numpy as np 
# Create splits 
N = len(df_no_duplicates)
N_train = int(N * 0.7)
N_train_eval = int(N * 0.1)
N_test = int(N * 0.1)
N_final_eval = N - N_train - N_train_eval - N_test 

assert N_final_eval > 0


indices = df_no_duplicates.index.to_series()
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
        "train"         : df_no_duplicates.loc[indices_train      , :],
        "train_eval"    : df_no_duplicates.loc[indices_train_eval , :],
        "test"          : df_no_duplicates.loc[indices_test       , :],
        "final_test"     : df_no_duplicates.loc[indices_final_test  , :],
    })
    .reset_index()
    .drop(columns=["level_1"])
    .rename(columns = {"level_0": "split"})
)
df_split.to_csv("./data/GWStance_preprocessed.csv", index = False)

from transformers import AutoModelForSequenceClassification

labels = list(df_split["label_text"].unique())
num_labels = len(labels)
id2label = {id:label for id, label in enumerate(labels)}
label2id = {label:id for id, label in enumerate(labels)}

MODEL_NAME = "google-bert/bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,                                        
).to(device="cpu")

print(model)

from transformers import AutoConfig
print(AutoConfig.from_pretrained(MODEL_NAME))


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


from datasets import DatasetDict, Dataset
from torch import Tensor

# Create a dataset from the splits we created before
grouped_ds_split = df_split.groupby("split")
dsd = DatasetDict({
    split :( 
        Dataset
        .from_pandas(grouped_ds_split.get_group(split))
        .with_format("torch", device="cpu")#, dtype=int)
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
    tokenized_entry = tokenizer(row["sentence-preprocessed"], **tokenizer_parameters)
    id_label = int(label2id[row["label_text"]])
    id_label_as_tensor = Tensor([int(i == id_label) for i in range(num_labels)])
    return {
        **row.copy(),
        "labels": id_label_as_tensor,
        "attention_mask" : tokenized_entry["attention_mask"].reshape(-1),
        "input_ids" : tokenized_entry["input_ids"].reshape(-1)
    }


dsd = dsd.map(preprocess_dataset, batch_size=32)
print(dsd)
dsd.save_to_disk("./outputs/dataset-dict")


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
base_model_output = model.base_model(**model_input)
classif_model_output = model(**model_input)
print(f'''
# model input keys: {', '.join(model_input)}
model input shape (pytorch tensor): {model_input["input_ids"].shape}
base model output keys: {', '.join(base_model_output)}
base model output last_hidden_state shape (pytorch tensor): {base_model_output.last_hidden_state.shape}
classification model output key: {', '.join(classif_model_output)}
classification model output logits shape (pytorch tensor): {classif_model_output.logits.shape}
''')

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding




#     save_strategy = "epoch",
#     save_total_limit = 5 + 1,

#     disable_tqdm = False,

#     use_cpu = True
# )

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


trainer = Trainer(
    model = model, 
    args = training_arguments,
    train_dataset=dsd["train"],
    eval_dataset=dsd["train_eval"],
    compute_metrics = compute_metrics
)

trainer.train()
# WARNING MODEL switches to mps need to handle that