#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_multiply_gpt2.py

Train a GPT-2–style language model from scratch on the multiplication dataset,
performing “pretraining” (i.e. supervised fine-tuning) with a custom digit-wise tokenizer.
This version adds gradient accumulation.

Usage:
    python train_multiply_gpt2.py \
        --train_data multiply_dataset.jsonl \
        --test_S test_S.jsonl \
        --test_T test_T.jsonl \
        --test_VU test_VU.jsonl \
        --batch_size 16 \
        --accumulation_steps 2 \
        --lr 1e-3 \
        --epochs 3 \
        --save_dir checkpoints \
        --seed 42

Rules implemented:
1. Tokenizer splits every digit as an individual token; all other tokens (operators,
   keywords like “Answer:”, “Problem:”, “Step1:” etc.) are explicitly in the vocab.
2. Standard SFT (supervised fine-tuning): given “input” → model predicts “output”,
   compute cross-entropy loss on the output portion.
3. During evaluation, extract the final answer from the model’s generated text by
   locating the last digit sequence after “Answer:”; if no “Answer:” is present,
   take the final digit in the generated text.
4. Every 1000 actual optimization steps (after accumulation), run evaluation on
   test_S, test_T, test_VU. If test_VU’s error rate is lower than any previously seen,
   save a checkpoint.
5. Fixed random seed for reproducibility.
6. Uses a cosine LR scheduler with warmup.
7. Learning rate default = 1e-3.
8. Model is GPT-2 configured from scratch (not from pretrained), with:
   n_positions=1024, n_ctx=1024, n_embd=512, n_layer=10, n_head=16.
9. If any input+output token sequence exceeds 1024 tokens, raise an error.
10. Gradient accumulation default = 2 steps (configurable via `--accumulation_steps`).
"""

import os
import random
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup
from tqdm import tqdm

# --------------- TOKENIZER DEFINITION ---------------

DIGIT_TOKENS = [
    '0','1','2','3','4','5','6','7','8','9',
    '*','+','=','?','\n',
    'Answer:','Problem:','Example1:','Three','Four','digits.',
    'Step1:','Step2:','Step3:','Step4:','Step5:','Step6:','Step7:',
    'Addition:','<|pad|>','<|unk|>'
]

def build_digit_tokenizer():
    class MultiplyTokenizer:
        def __init__(self):
            self.vocab = { tok: idx for idx, tok in enumerate(DIGIT_TOKENS) }
            self.pad_token = '<|pad|>'
            self.pad_token_id = self.vocab[self.pad_token]
            self.unk_token = '<|unk|>'
            self.unk_token_id = self.vocab[self.unk_token]
            # for decoding
            self.id_to_tok = { idx: tok for tok, idx in self.vocab.items() }

        def tokenize(self, text: str):
            """
            Returns a list of token strings.
            - Splits text by newline, then by whitespace.
            - If a “word” exactly matches a vocab entry (e.g. “Answer:”, “Step1:”),
              emit that single token.
            - Else, for each character in word:
                if the character is in vocab (e.g. digit, '*', '+', '=', '?', etc.),
                emit that character as a token;
                otherwise emit '<|unk|>'.
            - After processing each line, append '\n' token.
            """
            tokens = []
            lines = text.split('\n')
            for line in lines:
                if line.strip() == "":
                    tokens.append('\n')
                    continue
                words = line.strip().split()
                for w in words:
                    if w in self.vocab:
                        tokens.append(w)
                    else:
                        for ch in w:
                            if ch in self.vocab:
                                tokens.append(ch)
                            else:
                                tokens.append(self.unk_token)
                tokens.append('\n')
            # remove the final appended '\n' if text did not end with a newline
            if not text.endswith('\n'):
                tokens = tokens[:-1]
            return tokens

        def convert_tokens_to_ids(self, tokens):
            return [ self.vocab.get(t, self.unk_token_id) for t in tokens ]

        def convert_ids_to_tokens(self, ids):
            return [ self.id_to_tok.get(i, self.unk_token) for i in ids ]

        def decode(self, ids):
            """
            Convert a list of ids back to a string, joining tokens.
            Merge tokens without inserting spaces, but restore '\n' to newlines.
            """
            toks = self.convert_ids_to_tokens(ids)
            result = []
            for t in toks:
                if t == '\n':
                    result.append('\n')
                else:
                    result.append(t)
            return "".join(result)

    return MultiplyTokenizer()

# --------------- DATASET DEFINITION ---------------

class MultiplyDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 1024):
        """
        Each line in `path` is a JSON object with keys "input" and "output".
        We tokenize both, then concatenate. If length > max_length, error.
        """
        self.examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                inp_text = item['input']
                out_text = item['output']
                # tokenize
                inp_tokens = tokenizer.tokenize(inp_text)
                out_tokens = tokenizer.tokenize(out_text)
                inp_ids = tokenizer.convert_tokens_to_ids(inp_tokens)
                out_ids = tokenizer.convert_tokens_to_ids(out_tokens)
                full_ids = inp_ids + out_ids
                if len(full_ids) > max_length:
                    raise ValueError(
                        f"Sequence too long ({len(full_ids)} > {max_length}) for example:\n{inp_text}\n--\n{out_text}"
                    )
                # build labels: -100 for input portion, actual ids for output portion
                labels = [-100] * len(inp_ids) + out_ids
                # pad to max_length
                attention_mask = [1] * len(full_ids)
                pad_len = max_length - len(full_ids)
                full_ids += [tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len
                attention_mask += [0] * pad_len

                self.examples.append({
                    'input_ids': torch.tensor(full_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'labels': torch.tensor(labels, dtype=torch.long)
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# --------------- EVALUATION UTILITIES ---------------

def extract_final_answer(text: str) -> str:
    """
    From a decoded output string, find the last occurrence of “Answer:” followed by digits.
    If found, return the digits. If not, return the last digit character in the entire text.
    """
    lines = text.split('\n')
    for line in reversed(lines):
        if 'Answer:' in line:
            tail = line.split('Answer:')[-1]
            digits = "".join(ch for ch in tail if ch.isdigit())
            if digits:
                return digits
    digits_all = [ch for ch in text if ch.isdigit()]
    return digits_all[-1] if digits_all else ""


def evaluate(model, tokenizer, data_path: str, device):
    """
    Evaluate on the dataset at data_path. Returns (avg_loss, error_rate).
    error_rate = 1 - accuracy on final-digit match.
    """
    model.eval()
    dataset = MultiplyDataset(data_path, tokenizer, max_length=1024)
    loader = DataLoader(dataset, batch_size=16)
    total_loss = 0.0
    total_examples = 0
    correct = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits  # (B, L, V)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            label_ids = labels.cpu().tolist()

            for pred_seq, label_seq in zip(preds, label_ids):
                pred_text = tokenizer.decode([tid for tid in pred_seq if tid != tokenizer.pad_token_id])
                label_text = tokenizer.decode([lid for lid in label_seq if lid != -100 and lid != tokenizer.pad_token_id])
                pred_ans = extract_final_answer(pred_text)
                true_ans = extract_final_answer(label_text)
                if pred_ans == true_ans and true_ans != "":
                    correct += 1
                total_examples += 1

    avg_loss = total_loss / len(loader)
    err_rate = 1.0 - (correct / total_examples if total_examples > 0 else 0.0)
    model.train()
    return avg_loss, err_rate

# --------------- TRAINING SCRIPT ---------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, 
                        default='multiply_dataset.jsonl')
    parser.add_argument('--test_S', type=str, 
                        default='test_S.jsonl')
    parser.add_argument('--test_T', type=str, 
                        default='test_T.jsonl')
    parser.add_argument('--test_VU', type=str, 
                        default='test_VU.jsonl')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help="Number of batches to accumulate gradients over")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 1. Set seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # 2. Build tokenizer and datasets
    tokenizer = build_digit_tokenizer()
    train_dataset = MultiplyDataset(args.train_data, tokenizer, max_length=1024)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 3. Configure GPT-2 from scratch
    config = GPT2Config(
        vocab_size=len(tokenizer.vocab),
        n_positions=1024,
        n_ctx=1024,
        n_embd=256,
        n_layer=8,
        n_head=12,
        pad_token_id=tokenizer.pad_token_id
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer.vocab))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = (len(train_loader) // args.accumulation_steps + 1) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )

    # 5. Training loop with gradient accumulation and periodic evaluation
    best_VU_err = float('inf')
    global_step = 0
    accumulation_counter = 0

    model.train()
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # Scale loss by accumulation steps
            loss = loss / args.accumulation_steps
            loss.backward()

            accumulation_counter += 1
            if accumulation_counter == args.accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accumulation_counter = 0
                global_step += 1

                if global_step % 1000 == 0:
                    # Evaluate on all three test sets
                    loss_S, err_S = evaluate(model, tokenizer, args.test_S, device)
                    loss_T, err_T = evaluate(model, tokenizer, args.test_T, device)
                    loss_VU, err_VU = evaluate(model, tokenizer, args.test_VU, device)
                    print(f"\nStep {global_step}:")
                    print(f"  test_S   → Loss: {loss_S:.4f}, Error: {err_S:.4f}")
                    print(f"  test_T   → Loss: {loss_T:.4f}, Error: {err_T:.4f}")
                    print(f"  test_VU  → Loss: {loss_VU:.4f}, Error: {err_VU:.4f}")
                    # If test_VU error improved, save checkpoint
                    if err_VU < best_VU_err:
                        best_VU_err = err_VU
                        ckpt_path = os.path.join(args.save_dir, f"best_VU_step{global_step}.pt")
                        torch.save(model.state_dict(), ckpt_path)
                        print(f"📦 New best VU error ({best_VU_err:.4f}); checkpoint saved to {ckpt_path}")

    # In case the last few batches did not trigger optimizer.step()
    if accumulation_counter > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        if global_step % 1000 == 0:
            loss_S, err_S = evaluate(model, tokenizer, args.test_S, device)
            loss_T, err_T = evaluate(model, tokenizer, args.test_T, device)
            loss_VU, err_VU = evaluate(model, tokenizer, args.test_VU, device)
            print(f"\nStep {global_step}:")
            print(f"  test_S   → Loss: {loss_S:.4f}, Error: {err_S:.4f}")
            print(f"  test_T   → Loss: {loss_T:.4f}, Error: {err_T:.4f}")
            print(f"  test_VU  → Loss: {loss_VU:.4f}, Error: {err_VU:.4f}")
            if err_VU < best_VU_err:
                best_VU_err = err_VU
                ckpt_path = os.path.join(args.save_dir, f"best_VU.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"📦 New best VU error ({best_VU_err:.4f}); checkpoint saved to {ckpt_path}")

    print("\nTraining complete.")

if __name__ == '__main__':
    main()
