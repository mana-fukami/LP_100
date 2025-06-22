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

# 正解率を計算
print(f"{trainer.evaluate()}%")
"""
{'eval_loss': 0.2684279680252075, 'eval_accuracy': 0.9197247706422018, 'eval_runtime': 1.4505, 'eval_samples_per_second': 601.187, 'eval_steps_per_second': 9.652, 'epoch': 3.0}
"""
"""
{'loss': 0.236, 'grad_norm': 8.854281425476074, 'learning_rate': 6.666666666666667e-06, 'epoch': 1.0}
{'eval_loss': 0.23034018278121948, 'eval_accuracy': 0.911697247706422, 'eval_runtime': 1.5662, 'eval_samples_per_second': 556.767, 'eval_steps_per_second': 8.939, 'epoch': 1.0}
 33%|███████████████▋                               | 1053/3159 [04:23<07:38,  4.59it/s/home0/y2022/f2210543/.pyenv/versions/3.10.13/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
{'loss': 0.133, 'grad_norm': 4.726690769195557, 'learning_rate': 3.3333333333333333e-06, 'epoch': 2.0}
{'eval_loss': 0.2684279680252075, 'eval_accuracy': 0.9197247706422018, 'eval_runtime': 1.4541, 'eval_samples_per_second': 599.695, 'eval_steps_per_second': 9.628, 'epoch': 2.0}
 67%|███████████████████████████████▎               | 2106/3159 [08:57<03:51,  4.54it/s/home0/y2022/f2210543/.pyenv/versions/3.10.13/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
{'loss': 0.1003, 'grad_norm': 1.181188702583313, 'learning_rate': 0.0, 'epoch': 3.0}    
{'eval_loss': 0.27784305810928345, 'eval_accuracy': 0.9185779816513762, 'eval_runtime': 1.4594, 'eval_samples_per_second': 597.526, 'eval_steps_per_second': 9.593, 'epoch': 3.0}
{'train_runtime': 825.2973, 'train_samples_per_second': 244.817, 'train_steps_per_second': 3.828, 'train_loss': 0.15642503410549471, 'epoch': 3.0}                              
100%|███████████████████████████████████████████████| 3159/3159 [13:46<00:00,  3.82it/s]
/home0/y2022/f2210543/.pyenv/versions/3.10.13/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
100%|███████████████████████████████████████████████████| 14/14 [00:01<00:00, 10.52it/s]
{'eval_loss': 0.2684279680252075, 'eval_accuracy': 0.9197247706422018, 'eval_runtime': 1.4505, 'eval_samples_per_second': 601.187, 'eval_steps_per_second': 9.652, 'epoch': 3.0}
"""