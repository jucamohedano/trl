"""Replay published AlphaXiv parent turns with OpenEnv's existing helpers."""

import argparse
import ast
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.dataset as ds
from repl_env import LocalRLMRunner


TRACE_IDS = (
    "multi-2503.11576-q2-d0c4b43f",
    "multi-2504.09522-q4-46428352",
)
QUERY_PREFIX = "Extract verbatim text passages from the context that serve as evidence for the query: "
REPL_BLOCK = re.compile(r"```repl\s*(.*?)```", re.S)
PAPER_ID = re.compile(r"\d{4}\.\d{4,5}")
BATCH_LOOP = ast.parse("""for i in range(0, len(ids), 4):
    batch_ids = ids[i:i+4]
    batch_prompts = prompts[i:i+4]
    batch_contexts = [context[pid] for pid in batch_ids]
    batch_results = rlm_query_batched(batch_prompts, context_list=batch_contexts)
    results.append((batch_ids, batch_results))
""").body[0]


def value(node, assignments):
    """Read literal teacher inputs without executing their source code."""
    if isinstance(node, ast.Name):
        return value(assignments[node.id], assignments)
    if isinstance(node, ast.Subscript):
        return value(node.value, assignments)[ast.literal_eval(node.slice)]
    if isinstance(node, ast.List):
        return [value(item, assignments) for item in node.elts]
    if isinstance(node, ast.JoinedStr):
        return "".join(
            str(value(item.value if isinstance(item, ast.FormattedValue) else item, assignments))
            for item in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        if isinstance(node.right, ast.Call) and ast.unparse(node.right.func) == "get_paper_abstract":
            return value(node.left, assignments).split("\n\nPaper preview:", 1)[0].strip()
    try:
        return ast.literal_eval(node)
    except ValueError as error:
        raise ValueError(f"unsupported teacher literal: {type(node).__name__}") from error


def paper_contexts(node, assignments):
    if isinstance(node, ast.Name):
        return paper_contexts(assignments[node.id], assignments)
    if isinstance(node, ast.ListComp) and ast.unparse(node.elt) == "context[pid]":
        return value(node.generators[0].iter, assignments)
    if isinstance(node, ast.List) and all(isinstance(item, ast.Subscript) for item in node.elts):
        return [value(item.slice, assignments) for item in node.elts]
    raise ValueError("unsupported context_list expression")


def adapt(rows):
    assignments = {}
    paper_ids = []
    result_names = []
    root_turns = []
    direct_evidence = None
    skipped_turn_ids = []
    unpaired_paper_ids = []

    def render_batch(chosen, instructions, name):
        if len(chosen) != len(instructions) or not all(PAPER_ID.fullmatch(pid) for pid in chosen):
            raise ValueError("invalid child paper IDs or instruction count")
        paper_ids.extend(chosen)
        result_names.append(name)
        return f"""paper_ids = {chosen!r}
instructions = {instructions!r}
prompts = [
    f"Paper ID: {{pid}}\\n{{instruction}}\\nPaper text:\\n{{papers[pid]}}"
    for pid, instruction in zip(paper_ids, instructions)
]
{name} = rlm_query_batched(prompts)
print({name})"""

    for row in rows:
        statements = [stmt for block in REPL_BLOCK.findall(row["output_response"]) for stmt in ast.parse(block).body]
        if (
            not root_turns
            and not statements
            and re.fullmatch(r"\s*<think>\s*</think>\s*<\|im_end\|>\s*", row["output_response"])
        ):
            skipped_turn_ids.append(row["id"])
            continue
        commands = []
        words = []
        for stmt in statements:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                expr = stmt.value
                if name == "prompts" and ast.unparse(expr) == (
                    "[p + '\\n\\nPaper preview:\\n' + get_paper_abstract(context, pid) for p, pid in zip(prompts, ids)]"
                ):
                    # The child prompt already receives the full resolved paper.
                    continue
                assignments[name] = expr
                if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
                    continue
                if expr.func.id == "list_papers":
                    commands.append("import json\npapers = json.loads(context)\nprint(list(papers))")
                elif expr.func.id == "search":
                    words.append(ast.literal_eval(expr.args[1]))
                elif expr.func.id == "rlm_query_batched":
                    arguments = {kw.arg: kw.value for kw in expr.keywords}
                    chosen = paper_contexts(arguments["context_list"], assignments)
                    instructions = value(expr.args[0], assignments)
                    # SkyRL pairs prompts and contexts with zip().
                    unpaired_paper_ids.extend(chosen[len(instructions) :])
                    chosen = chosen[: len(instructions)]
                    commands.append(render_batch(chosen, instructions, name))
            elif isinstance(stmt, ast.For):
                if ast.dump(stmt) == ast.dump(BATCH_LOOP):
                    chosen = value(assignments["ids"], assignments)
                    instructions = value(assignments["prompts"], assignments)
                    if len(chosen) != len(instructions):
                        raise ValueError("invalid child paper IDs or instruction count")
                    for start in range(0, len(chosen), 4):
                        commands.append(
                            render_batch(
                                chosen[start : start + 4], instructions[start : start + 4], f"results_{start // 4}"
                            )
                        )
                elif any(
                    isinstance(node, ast.Call) and ast.unparse(node.func) == "rlm_query_batched"
                    for node in ast.walk(stmt)
                ):
                    raise ValueError("teacher batches children in a loop")
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if ast.unparse(stmt.value.func) == "FINAL_VAR":
                    if not result_names:
                        name = ast.literal_eval(stmt.value.args[0])
                        direct_evidence = value(assignments[name], assignments)
                        if not isinstance(direct_evidence, list) or not all(
                            isinstance(span, str) for span in direct_evidence
                        ):
                            raise ValueError("unsupported direct evidence answer")
                        commands.append(f'answer["content"] = {json.dumps(direct_evidence)!r}\nanswer["ready"] = True')
                    else:
                        commands.append(
                            f"""evidence = []
for batch in [{", ".join(result_names)}]:
    for result in batch:
        evidence.extend(json.loads(result))
answer["content"] = json.dumps(evidence)
answer["ready"] = True"""
                        )
        if words:
            commands.append(
                f"""for keyword in {words!r}:
    print(keyword, [pid for pid, paper in papers.items() if keyword.lower() in paper.lower()])"""
            )
        if not commands:
            raise ValueError(f"no supported actions in teacher turn {row['id']}")
        root_turns.append("\n".join(commands))
    if not root_turns[-1].endswith('answer["ready"] = True'):
        raise ValueError("missing final submission")
    return paper_ids, root_turns, direct_evidence, skipped_turn_ids, unpaired_paper_ids


def replay(rows, task):
    paper_ids, root_turns, direct_evidence, skipped_turn_ids, unpaired_paper_ids = adapt(rows)
    papers = {paper["paperId"]: paper["text"] for paper in task["papers"]}
    evidence = {item["paperId"]: [span["text"] for span in item["selections"]] for item in task["evidence"]}
    if not evidence:
        raise ValueError("task has no reference evidence")
    if not set(paper_ids) <= papers.keys():
        raise ValueError("teacher selected papers absent from the public task")
    delegated = {pid: spans for pid, spans in evidence.items() if pid in paper_ids}
    if any(not span or span not in papers[pid] for pid, spans in delegated.items() for span in spans):
        raise ValueError("a scripted child excerpt is absent from its public paper text")
    children = []
    parent_actions = iter(root_turns)

    def chat(messages, model=None):
        prompt = messages[-1]["content"]
        match = re.search(r"Paper ID: ([\d.]+)\n", prompt) if len(messages) == 2 else None
        if match:
            pid = match[1]
            assert pid in paper_ids and papers[pid] in prompt
            children.append(pid)
            code = f"answer['content'] = {json.dumps(evidence.get(pid, []))!r}\nanswer['ready'] = True"
        else:
            code = next(parent_actions)
        return f"```repl\n{code}\n```"

    query = task["question"].strip().strip('"')
    prompt = f"{query}\nReturn a JSON list of verbatim passages that answer the question."
    result = LocalRLMRunner(chat, max_depth=2, max_iterations=len(root_turns)).run(json.dumps(papers), prompt)
    expected = (
        direct_evidence
        if direct_evidence is not None
        else [span for pid in paper_ids for span in delegated.get(pid, [])]
    )
    if sorted(children) != sorted(paper_ids) or len(result.child_traces) != len(paper_ids):
        raise ValueError("OpenEnv did not execute every translated child call")
    if any(trace.error for trace in result.child_traces) or any(
        message["role"] == "user" and "\n\nERROR: " in message["content"] for message in result.messages
    ):
        raise ValueError("OpenEnv reported a REPL execution error")
    if result.final_answer is None or json.loads(result.final_answer) != expected:
        raise ValueError("OpenEnv final answer differs from the expected evidence")
    if sum(message["role"] == "assistant" for message in result.messages) != len(root_turns):
        raise ValueError("OpenEnv did not execute every translated parent turn")
    return {
        "source_paper_id": task["source_paper_id"],
        "source_turn_ids": [row["id"] for row in rows],
        "messages": result.messages,
    }, {
        "reference_paper_ids": sorted(evidence),
        "delegated_paper_ids": sorted(set(paper_ids)),
        "undelegated_reference_paper_ids": sorted(evidence.keys() - set(paper_ids)),
        "reference_excerpts_returned": len(set(expected) & {span for spans in evidence.values() for span in spans}),
        "reference_excerpt_count": len({span for spans in evidence.values() for span in spans}),
        "returned_excerpt_count": len(expected),
        "unverified_direct_excerpt_count": sum(
            not span or not any(span in paper for paper in papers.values()) for span in (direct_evidence or [])
        ),
        "skipped_empty_turn_ids": skipped_turn_ids,
        "unpaired_paper_ids": unpaired_paper_ids,
    }


def match_task(rows, candidates):
    questions = {
        row["input_prompt"].rsplit(QUERY_PREFIX, 1)[1].split("\nReturn a Python list", 1)[0]
        for row in rows
        if QUERY_PREFIX in row["input_prompt"]
    }
    matching = [task for task in candidates if task["question"].strip().strip('"') in questions]
    if len(matching) != 1:
        raise ValueError(f"expected one matching task, found {len(matching)}")
    return matching[0]


def measure(example, tokenizer, template):
    from itertools import groupby

    tokens = tokenizer.apply_chat_template(
        example["messages"],
        chat_template=template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids, mask = tokens["input_ids"], tokens["assistant_masks"]
    masked = [
        tokenizer.decode([token for token, _ in span])
        for selected, span in groupby(zip(ids, mask, strict=True), key=lambda item: item[1])
        if selected
    ]
    assistant = [message["content"] for message in example["messages"] if message["role"] == "assistant"]
    if len(masked) != len(assistant) or any(
        span.strip().removeprefix("<think>\n\n</think>").removeprefix("</think>").removesuffix("<|im_end|>").strip()
        != turn.strip()
        for span, turn in zip(masked, assistant, strict=True)
    ):
        raise ValueError("assistant mask does not match the adapted code turns")
    return {"tokens": len(ids), "assistant_tokens": sum(mask)}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="Audit and replay the whole parent release.")
    parser.add_argument("--tokenizer", type=Path, help="Local Qwen3.5-4B tokenizer directory.")
    args = parser.parse_args()

    groups = defaultdict(list)
    for row in map(json.loads, args.traces.read_text().splitlines()):
        trace_id, turn = row["id"].rsplit("-", 1)
        if "-child" not in trace_id and (args.all or trace_id in TRACE_IDS):
            groups[trace_id].append((int(turn), row))
    sources = {trace_id.split("-")[1] for trace_id in groups}
    tasks = defaultdict(list)
    for task in (
        ds.dataset(args.tasks, format="parquet")
        .to_table(filter=ds.field("source_paper_id").isin(sorted(sources)))
        .to_pylist()
    ):
        tasks[task["source_paper_id"]].append(task)

    tokenizer = template = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        from trl.chat_template_utils import get_training_chat_template

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        template = get_training_chat_template(tokenizer, require_prefix_preservation=False)

    examples = []
    report = {}
    for trace_id, turns in sorted(groups.items()):
        source = trace_id.split("-")[1]
        entry = report[trace_id] = {"source_paper_id": source, "turns": len(turns)}
        try:
            turns.sort()
            if [index for index, _ in turns] != list(range(1, len(turns) + 1)):
                raise ValueError("teacher turns are not contiguous")
            rows = [row for _, row in turns]
            example, replay_details = replay(rows, match_task(rows, tasks[source]))
            entry.update(replay_details)
            entry["replay"] = "pass"
            if tokenizer is not None:
                entry.update(measure(example, tokenizer, template))
            entry["status"] = "pass"
            examples.append((trace_id, example))
        except (
            AssertionError,
            AttributeError,
            IndexError,
            KeyError,
            StopIteration,
            SyntaxError,
            TypeError,
            ValueError,
        ) as error:
            if not args.all:
                raise
            entry["status"] = "fail"
            entry["reason"] = f"{type(error).__name__}: {error}"[:300]

    write_jsonl(args.output, [example for _, example in examples])
    if args.all:
        passing_sources = sorted({example["source_paper_id"] for _, example in examples})
        random.Random(42).shuffle(passing_sources)
        validation_sources = set(passing_sources[:2])
        for split in ("train", "validation"):
            split_examples = []
            for trace_id, example in examples:
                if (example["source_paper_id"] in validation_sources) == (split == "validation"):
                    report[trace_id]["split"] = split
                    split_examples.append(example)
            write_jsonl(args.output.with_suffix(f".{split}.jsonl"), split_examples)
        args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
