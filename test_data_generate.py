#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_three_testsets.py

Generates three JSONL test sets (each with 1000 input-output pairs) in the same
format as the training dataset:

1. test_S.jsonl   - both multiplicand and multiplier sampled from S = {100..999}
2. test_T.jsonl   - both sampled from T = {1000..9999 | x % 50 ∈ {2,3,5,7,11,13,17,19,21,22,24,25,26,27,28,30,32,33,34,35,36,38,39,40,41,43,47}}
3. test_VU.jsonl  - multiplicand from V = {1000..9999}, multiplier from U = {1000..9999 | x % 50 ∈ {1,4,6,8,9,10,12,14,15,16,18,20,23,29,31,37,42,44,45,46,48,49,0}}

Each JSON object has:
    "input":  a 1-shot prompt (random demonstration example + “Problem: a*b=?”)
    "output": the detailed, step-by-step solution for the problem (ending with "Answer:<product>")

Usage:
    python generate_three_testsets.py

Outputs:
    test_S.jsonl
    test_T.jsonl
    test_VU.jsonl
"""

import random
import json
import sys

def digits_list(x):
    """
    Return a list of decimal digits of x, most significant first.
    E.g. digits_list(987) -> [9, 8, 7]
    """
    return [int(d) for d in str(x)]

def long_mul_steps(a: int, b: int) -> str:
    """
    Compute the long‐multiplication steps for a * b, including:
      - Step1: single product “d_i*d_j=<val>”
      - Steps 2..n:
          StepK:
          term1+term2+...+carry_prev
          =v1+v2+...+carry_prev
          Addition:
          <ones‐column digits + carry_in>=<sum zero‐padded>
          <tens‐column digits + carry_from_ones>=<sum zero‐padded>
          ...
          <expanded_terms>=<partial_sum>
    Finally append “Answer:<product>” on its own line.
    Returns a multiline string with '\n'.
    """
    A = digits_list(a)
    B = digits_list(b)
    Ar = list(reversed(A))   # least significant digit first
    Br = list(reversed(B))
    lenA = len(Ar)
    lenB = len(Br)
    num_steps = lenA + lenB - 1

    lines = []
    # Step1: only one term Ar[0] * Br[0]
    prod = Ar[0] * Br[0]
    carry_prev = prod // 10
    lines.append("Step1:")
    lines.append(f"{Ar[0]}*{Br[0]}={prod}")

    # Steps 2..num_steps
    for i in range(1, num_steps):
        term_exprs = []
        val_list = []
        j_min = max(0, i - (lenB - 1))
        j_max = min(i, lenA - 1)
        # collect all products Ar[j] * Br[i - j], from highest j down to lowest
        for j in range(j_max, j_min - 1, -1):
            term_exprs.append(f"{Ar[j]}*{Br[i - j]}")
            val_list.append(Ar[j] * Br[i - j])
        # include the previous carry as the last term
        term_exprs.append(str(carry_prev))
        val_list.append(carry_prev)

        partial_sum = sum(val_list)
        carry_curr = partial_sum // 10

        lines.append(f"Step{i+1}:")
        lines.append("+".join(term_exprs))
        expanded_terms = "+".join(str(v) for v in val_list)
        lines.append(f"={expanded_terms}")

        # Addition breakdown (digit-by-digit with carry propagation)
        lines.append("Addition:")
        width = max(len(str(v)) for v in val_list)  # number of digits in largest term
        val_strs = [str(v).zfill(width) for v in val_list]
        carry = 0
        for col in range(width):
            # extract digit in column `col` (0 = ones, 1 = tens, etc.)
            digits_col = [int(vs[-1-col]) for vs in val_strs]
            sum_col = sum(digits_col) + carry
            sum_str = str(sum_col).zfill(width)
            row_terms = digits_col + [carry]
            row_line = "+".join(str(d) for d in row_terms) + f"={sum_str}"
            lines.append(row_line)
            carry = sum_col // 10

        # final line of this step: show the expanded terms = partial_sum (no padding)
        lines.append(f"{expanded_terms}={partial_sum}")

        carry_prev = carry_curr

    # Append final answer
    product = a * b
    lines.append(f"Answer:{product}")

    return "\n".join(lines)

def build_one_shot_prompt(a: int, b: int, ex_a: int, ex_b: int) -> (str, str):
    """
    Build the 1-shot “input” (with demonstration example ex_a * ex_b plus its answer)
    and the full “output” for a single problem (a, b). Always label “Three digits.”
    or “Four digits.” based on the length of ex_a/ex_b (they must match the problem's length).
    Returns (input_str, output_str), each with internal '\n'.
    """
    # Determine label by checking digit‐length of ex_a
    label = "Three digits." if 100 <= ex_a <= 999 else "Four digits."

    # Demonstration example steps (including its Answer line)
    ex_steps = long_mul_steps(ex_a, ex_b)

    # Build the input string
    lines = []
    lines.append("Example1:")
    lines.append(f"{ex_a}*{ex_b}=?")
    lines.append(label)
    lines.append(ex_steps)
    lines.append("")         # blank line before “Problem:”
    lines.append("Problem:")
    lines.append(f"{a}*{b}=?")

    input_str = "\n".join(lines)
    output_str = long_mul_steps(a, b)
    return input_str, output_str

def main():
    random_seed = 42
    random.seed(random_seed)

    # Define sets S, T, U, V
    S = list(range(100, 1000))
    allowed_T = {2,3,5,7,11,13,17,19,21,22,24,25,26,27,28,30,32,33,34,35,36,38,39,40,41,43,47}
    T = [x for x in range(1000, 10000) if (x % 50) in allowed_T]

    allowed_U = {1,4,6,8,9,10,12,14,15,16,18,20,23,29,31,37,42,44,45,46,48,49,0}
    U = [x for x in range(1000, 10000) if (x % 50) in allowed_U]

    V = list(range(1000, 10000))

    # Each testset has 1000 examples
    n_examples = 1000

    try:
        fout_S = open("test_S.jsonl", "w", encoding="utf-8")
        fout_T = open("test_T.jsonl", "w", encoding="utf-8")
        fout_VU = open("test_VU.jsonl", "w", encoding="utf-8")
    except Exception as e:
        print(f"Error: cannot open output file: {e}", file=sys.stderr)
        sys.exit(1)

    # 1) test_S.jsonl: both multiplicand and multiplier from S; demonstration also from S
    for _ in range(n_examples):
        ex_a = random.choice(S)
        ex_b = random.choice(S)
        a = random.choice(S)
        b = random.choice(S)
        input_str, output_str = build_one_shot_prompt(a, b, ex_a, ex_b)
        record = {"input": input_str, "output": output_str}
        fout_S.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2) test_T.jsonl: both from T; demonstration also from T
    for _ in range(n_examples):
        ex_a = random.choice(T)
        ex_b = random.choice(T)
        a = random.choice(T)
        b = random.choice(T)
        input_str, output_str = build_one_shot_prompt(a, b, ex_a, ex_b)
        record = {"input": input_str, "output": output_str}
        fout_T.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 3) test_VU.jsonl: multiplicand from V, multiplier from U;
    #    demonstration ex_a from V, ex_b from U
    for _ in range(n_examples):
        ex_a = random.choice(V)
        ex_b = random.choice(U)
        a = random.choice(V)
        b = random.choice(U)
        input_str, output_str = build_one_shot_prompt(a, b, ex_a, ex_b)
        record = {"input": input_str, "output": output_str}
        fout_VU.write(json.dumps(record, ensure_ascii=False) + "\n")

    fout_S.close()
    fout_T.close()
    fout_VU.close()
    print("Generated:")
    print("  • test_S.jsonl  (S × S, 1000 examples)")
    print("  • test_T.jsonl  (T × T, 1000 examples)")
    print("  • test_VU.jsonl (V × U, 1000 examples)")

if __name__ == "__main__":
    main()
