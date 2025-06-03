#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multiply_dataset_generator.py

Generates a JSONL dataset for 3-digit and 4-digit long‐multiplication tasks,
including 1-shot prompts with fully detailed, per‐step arithmetic (including
“Addition:” breakdowns) and answers. The first 1,000,000 examples are 3‐digit
multiplications; the next n examples are “rare” 4‐digit multiplications under
a specified modulo‐50 remainder constraint.

Usage:
    python multiply_dataset_generator.py [--n N] [--output FILE]

Defaults:
    N = 1000
    FILE = multiply_dataset.jsonl

Each line in the output file is a JSON object with two keys:
    "input":  the full 1-shot prompt (random demonstration example with answer +
              “Problem: … =?”)
    "output": the fully detailed solution steps for the problem (ending with "Answer:<product>")
"""

import argparse
import json
import random
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a JSONL dataset of long‐multiplication examples."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of 4-digit examples to generate (default: 1000)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="multiply_dataset.jsonl",
        help="Output JSONL filename (default: multiply_dataset.jsonl)."
    )
    return parser.parse_args()

def digits_list(x):
    """
    Return a list of decimal digits of x, most significant first.
    e.g. digits_list(987) -> [9, 8, 7]
    """
    return [int(d) for d in str(x)]

def long_mul_steps(a: int, b: int) -> str:
    """
    Compute the long‐multiplication steps for a * b, mirroring the style:
      - Step1: single product “d_i*d_j=<val>”
      - Subsequent steps:
          StepK:
          term1+term2+…+carry_prev
          =val1+val2+…+carry_prev
          Addition:
          <ones‐column digits + carry_in>=<sum at ones, zero‐padded>
          <tens‐column digits + carry_from_ones>=<sum at tens, zero‐padded>
          …
          <full expanded terms>=<partial_sum>

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

def build_one_shot_prompt(a: int, b: int, example_pairs: tuple, is_four_digit: bool) -> (str, str):
    """
    Build the 1-shot “input” and full “output” for a single example (a, b).
    example_pairs is (ex_a, ex_b), drawn randomly from S or T.
    If is_four_digit is False, we label “Three digits.”; else “Four digits.”
    Returns (input_str, output_str), each with internal '\n' as needed.
    """
    ex_a, ex_b = example_pairs
    ex_steps = long_mul_steps(ex_a, ex_b)  # includes "Answer:<ex_a*ex_b>" at the end
    label = "Four digits." if is_four_digit else "Three digits."

    lines = []
    lines.append("Example1:")
    lines.append(f"{ex_a}*{ex_b}=?")
    lines.append(label)
    lines.append(ex_steps)
    lines.append("")         # blank line before “Problem:”
    lines.append("Problem:")
    lines.append(f"{a}*{b}=?")

    input_str = "\n".join(lines)
    # For the “output”, provide full steps + "Answer:<a*b>"
    output_str = long_mul_steps(a, b)
    return input_str, output_str

def main():
    args = parse_args()
    n_four = args.n
    out_file = args.output

    random_seed = 42
    random.seed(random_seed)

    # Build the “rare” set T for 4-digit: 1000..9999 where x % 50 in the specified list
    allowed_remainders = {
        2,3,5,7,11,13,17,19,21,22,24,25,26,27,28,30,32,33,34,35,36,38,39,40,41,43,47
    }
    four_candidates = [x for x in range(1000, 10000) if (x % 50) in allowed_remainders]

    total_three = 1_000_000
    total_four = n_four

    try:
        fout = open(out_file, "w", encoding="utf-8")
    except Exception as e:
        print(f"Error: cannot open output file {out_file}: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate 3-digit examples
    for _ in range(total_three):
        # Sample a random demonstration example (ex_a, ex_b) from S = 100..999
        ex_a = random.randint(100, 999)
        ex_b = random.randint(100, 999)
        # Sample the actual problem (a, b) also from 100..999
        a = random.randint(100, 999)
        b = random.randint(100, 999)

        input_str, output_str = build_one_shot_prompt(a, b, (ex_a, ex_b), is_four_digit=False)
        record = {"input": input_str, "output": output_str}
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Generate 4-digit examples
    for _ in range(total_four):
        # Sample a random demonstration example (ex_a, ex_b) from T (four_candidates)
        ex_a = random.choice(four_candidates)
        ex_b = random.choice(four_candidates)
        # Sample the actual problem (a, b) also from four_candidates
        a = random.choice(four_candidates)
        b = random.choice(four_candidates)

        input_str, output_str = build_one_shot_prompt(a, b, (ex_a, ex_b), is_four_digit=True)
        record = {"input": input_str, "output": output_str}
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    fout.close()
    print(f"Finished generating {total_three + total_four} examples → {out_file}")

if __name__ == "__main__":
    main()
