#!/usr/bin/env python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollator, Trainer, TrainingArguments
from datasets import Dataset, load_metric, load_dataset, Features, Value
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score
import argparse


def tokenize_function(s):
    seq_split = " ".join(s['seq'])
    return tokenizer(seq_split)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    specificity = recall_score(labels, preds, pos_label=0)
    return {
        'accuracy': acc,
        'recall (TPR)': recall,
        'specificity (TNR)': specificity,
        'f1': f1,
        'precision': precision
    }


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-GA", help="gradient accumulation", type=int, required=True)
parser.add_argument("-WD", help="weight decay", type=float, required=True)
parser.add_argument("-LR", help="learning rate", type=float, required=True)
args = parser.parse_args()

HF_MODEL = "Rostlab/prot_bert_bfd"
MODEL_SAVE = f"M1_ProtBertBFD_{args.GA}_{args.WD}_{args.LR}"
filelog = open('log.txt', 'a')
filelog.write(f"Model with params: GA = {args.GA}, WD = {args.WD}, LR = {args.LR}\n")

# load dataset, split train dataset into new train + test (for us validation)
dss = load_dataset('EvaKlimentova/knots_AF')
dss = dss.rename_column("uniprotSequence", "seq")
dss = dss['train'].train_test_split(test_size=0.1, seed=42, shuffle=True)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, do_lower_case=False)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)

tokenized_datasets = dss.map(tokenize_function,
                             remove_columns=['seq', 'ID', 'latestVersion', 'globalMetricValue', 'uniprotStart',
                                             'uniprotEnd', 'Length', 'Domain_architecture', 'InterPro', 'Max_Topology',
                                             'Max Freq', 'Knot Core', 'FamilyName'], num_proc=4)
tokenized_datasets.set_format("pt")

training_args = TrainingArguments(MODEL_SAVE, learning_rate=args.LR, warmup_ratio=0.1, lr_scheduler_type='cosine',
                                  fp16=True,
                                  evaluation_strategy="epoch", save_strategy="epoch", per_device_train_batch_size=1,
                                  per_device_eval_batch_size=1, gradient_accumulation_steps=args.GA,
                                  num_train_epochs=1, load_best_model_at_end=True, save_total_limit=1,
                                  weight_decay=args.WD, report_to='none', gradient_checkpointing=True,
                                  optim="adafactor")

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],   # for debugging .select(range(10)), 
    eval_dataset=tokenized_datasets["test"],   # for debugging.select(range(10)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(MODEL_SAVE)
metrics = trainer.evaluate()
filelog.write(f"Accuracy {round(metrics['eval_accuracy'], 4)}, TPR {round(metrics['eval_recall (TPR)'], 4)}, TNR "
              f"{round(metrics['eval_specificity (TNR)'], 4)}, f1 {round(metrics['eval_f1'], 4)}, Precision {round(metrics['eval_precision'], 4)}\n\n")
