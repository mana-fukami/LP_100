"""
訓練セットを用い、事前学習済みモデルを極性分析タスク向けにファインチューニングせよ。
検証セット上でファインチューニングされたモデルの正解率を計測せよ。
"""
from lesson85 import train_dataset,dev_dataset,tokenizer
from lesson86 import CustomCollate
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification,Trainer,TrainingArguments,EarlyStoppingCallback
from sklearn.metrics import accuracy_score

# GPUに移動
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# 事前学習済みモデルの読み込み
model=AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2).to(device)
model=nn.DataParallel(model)
model=model.to(device)

# Trainerの設定
training_args=TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    label_names=["labels"],
    lr_scheduler_type="linear",
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    remove_unused_columns=False,
    report_to="none"
)

# 評価指標
def compute_metrics(eval_pred):
    logits,labels=eval_pred
    preds=torch.argmax(torch.tensor(logits),dim=1)
    return {"accuracy": accuracy_score(labels,preds)}

# Trainerで学習
trainer=Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=CustomCollate,
    compute_metrics=compute_metrics,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

# モデルの評価
model.eval()
correct=0
total=0

dev_dataloader=DataLoader(dev_dataset,batch_size=64,shuffle=True, collate_fn=CustomCollate)
with torch.no_grad(): #計算の高速化のために勾配計算を無効にする
    for batch in dev_dataloader:
        features=batch["input_ids"]
        labels=batch["labels"].to(device).view(-1,1)

        # モデルの出力を取得
        outputs=model(features, attention_mask=batch["attention_mask"].to(device))
        logits = outputs.logits

        # 0.5を閾値として二値分類
        preds = torch.argmax(logits, dim=1)

# 正解率を計算
print(f"正解率: {accuracy_score(labels,preds)}%")
"""
正解率: 0.925%
"""
"""
/home0/y2022/f2210543/.pyenv/versions/3.10.13/lib/python3.10/site-packages/huggingface_hub/file_download.py:943: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Using cpu device
/home0/y2022/f2210543/.pyenv/versions/3.10.13/lib/python3.10/site-packages/huggingface_hub/file_download.py:943: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
  0%|                                                                                                   | 0/6315 [00:00<?, ?it/s]/home0/y2022/f2210543/.pyenv/versions/3.10.13/lib/python3.10/site-packages/torch/utils/data/dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 0.2296, 'grad_norm': 8.906991958618164, 'learning_rate': 6.666666666666667e-06, 'epoch': 1.0}                           
{'eval_loss': 0.23790372908115387, 'eval_accuracy': 0.9105504587155964, 'eval_runtime': 22.7996, 'eval_samples_per_second': 38.246, 'eval_steps_per_second': 1.228, 'epoch': 1.0}                                                                                 
 33%|████████████████████████████                                                        | 2105/6315 [1:37:35<2:58:03,  2.54s/it/home0/y2022/f2210543/.pyenv/versions/3.10.13/lib/python3.10/site-packages/torch/utils/data/dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 0.1269, 'grad_norm': 2.5534024238586426, 'learning_rate': 3.3333333333333333e-06, 'epoch': 2.0}                         
{'eval_loss': 0.2763686180114746, 'eval_accuracy': 0.9185779816513762, 'eval_runtime': 22.7488, 'eval_samples_per_second': 38.332, 'eval_steps_per_second': 1.231, 'epoch': 2.0}                                                                                  
 67%|████████████████████████████████████████████████████████                            | 4210/6315 [3:15:13<1:18:09,  2.23s/it/home0/y2022/f2210543/.pyenv/versions/3.10.13/lib/python3.10/site-packages/torch/utils/data/dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 0.0922, 'grad_norm': 1.0613242387771606, 'learning_rate': 0.0, 'epoch': 3.0}                                            
{'eval_loss': 0.30975887179374695, 'eval_accuracy': 0.9185779816513762, 'eval_runtime': 22.784, 'eval_samples_per_second': 38.272, 'eval_steps_per_second': 1.229, 'epoch': 3.0}                                                                                  
{'train_runtime': 17582.5773, 'train_samples_per_second': 11.491, 'train_steps_per_second': 0.359, 'train_loss': 0.14959282591903578, 'epoch': 3.0}                                                                                                               
100%|██████████████████████████████████████████████████████████████████████████████████████| 6315/6315 [4:53:02<00:00,  2.78s/it]
正解率: 0.925%
"""