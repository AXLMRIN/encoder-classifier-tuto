from datasets import Dataset, DatasetDict
import numpy as np 
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class PredictionPipe:
    def __init__(
        self, 
        checkpoint_dir : str, 
        model_name : str,
        text_column : str, 
        labels : list[str], 
        tokenizer_parameters : dict,
        device_batch_size : int, 
        device : str | None = None,
        )->None:
        '''
        '''
        self.__device = device if device is not None else "cpu"
        self.__device_batch_size = device_batch_size
        self.__text_column = text_column
        self.__labels = labels
        self.__tokenizer_parameters = tokenizer_parameters

        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir, 
        ).to(device=self.__device)

    def predict(self, dsd : DatasetDict, splits : list[str] = ["test"]) ->  dict[str:pd.DataFrame]:
        output = {}
        for split in splits: 
            ds = dsd[split].with_format("torch", device=self.__device,)
            local_id = []
            local_logits = []
            for batch in ds.batch(self.__device_batch_size):
                local_id += batch["id"]

                texts = batch[self.__text_column]
                logits : np.ndarray = (
                    self.__model(**self.__tokenizer(texts, **self.__tokenizer_parameters))
                    .logits
                    .detach()
                    .numpy()
                )
                local_logits += logits.tolist()
            output[split] = pd.DataFrame(local_logits, index = local_id, columns = self.__labels)
        return output
    