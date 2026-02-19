from transformers import AutoModelForSequenceClassification
from datasets import load_from_disk 
import numpy as np 
import pandas as pd 
DEVICE = "cpu"

dsd = load_from_disk("./outputs/dataset-dict")
for split in dsd:
    dsd[split] = dsd[split].with_format("torch", device="cpu")
print(dsd)
id2label = {0: 'agrees', 1: 'neutral', 2: 'disagrees'}

model = AutoModelForSequenceClassification.from_pretrained(
    "./models/training/checkpoint-90"
)
model = model.to(device=DEVICE)

labels_true : list[int] = []
labels_pred : list[int] = []

for batch in dsd["test"].batch(batch_size=16, drop_last_batch=False):
    
    model_input = {
        'input_ids' : batch['input_ids'],
        'attention_mask' : batch['attention_mask']
    }

    logits : np.ndarray = model(**model_input).logits.detach().numpy()
    
    batch_of_true_label = [id2label[np.argmax(row).item()] for row in batch["labels"]]
    labels_true.extend(batch_of_true_label)

    batch_of_pred_label = [id2label[np.argmax(row).item()] for row in logits]
    labels_pred.extend(batch_of_pred_label)


(
    pd.DataFrame({
        "predict" : labels_pred, 
        "gold_standard": labels_true
    })
    .to_csv("./outputs/prediction.csv", index = False)
)