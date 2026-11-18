"""Replay two public AlphaXiv parent traces with OpenEnv's existing helpers."""

import argparse
import ast
import json
import re
from pathlib import Path

import pyarrow.dataset as ds
from repl_env import LocalRLMRunner


TRACE_IDS = (
    "multi-2503.11576-q2-d0c4b43f",
    "multi-2504.09522-q4-46428352",
)
QUERY_PREFIX = "Extract verbatim text passages from the context that serve as evidence for the query: "


def adapt(rows):
    blocks = [ast.parse(re.search(r"```repl\s*(.*?)```", row["output_response"], re.S)[1]) for row in rows]
    query = rows[1]["input_prompt"].rsplit(QUERY_PREFIX, 1)[1].split("\nReturn a Python list", 1)[0]
    root_turns = []
    for i, block in enumerate(blocks[:2]):
        words = [
            ast.literal_eval(call.args[1])
            for statement in block.body
            if isinstance(statement, ast.Assign)
            and isinstance(call := statement.value, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "search"
        ]
        assert len(words) == 2
        setup = "import json\npapers = json.loads(context)\nprint(list(papers))\n" if i == 0 else ""
        root_turns.append(
            setup
            + f"""for keyword in {words!r}:
    print(keyword, [pid for pid, paper in papers.items() if keyword.lower() in paper.lower()])"""
        )

    assignments = {stmt.targets[0].id: stmt.value for stmt in blocks[2].body}
    paper_ids = ast.literal_eval(assignments["ids1"])
    instructions = [
        ast.literal_eval(item.left).split("\n\nPaper preview:", 1)[0] for item in assignments["prompts1"].elts
    ]
    assert len(paper_ids) == len(instructions)
    assert ast.unparse(blocks[3].body[-1]) == "FINAL_VAR('evidence')"
    root_turns.extend(
        [
            f"""paper_ids = {paper_ids!r}
instructions = {instructions!r}
prompts = [
    f"Paper ID: {{pid}}\\n{{instruction}}\\nPaper text:\\n{{papers[pid]}}"
    for pid, instruction in zip(paper_ids, instructions)
]
results = rlm_query_batched(prompts)
print(results)""",
            """evidence = []
for result in results:
    evidence.extend(json.loads(result))
answer = json.dumps(evidence)
print(FINAL_VAR("answer"))""",
        ]
    )
    return query, paper_ids, root_turns


def replay(rows, tasks):
    query, paper_ids, root_turns = adapt(rows)
    matches = [row for row in tasks if row["question"].strip().strip('"') == query]
    assert len(matches) == 1, query
    task = matches[0]
    papers = {paper["paperId"]: paper["text"] for paper in task["papers"]}
    evidence = {item["paperId"]: [span["text"] for span in item["selections"]] for item in task["evidence"]}
    assert evidence and evidence.keys() <= set(paper_ids)
    assert all(span in papers[pid] for pid, spans in evidence.items() for span in spans)
    children = []
    parent_actions = iter(root_turns)

    def chat(messages, model=None):
        prompt = messages[-1]["content"]
        match = re.search(r"Paper ID: ([\d.]+)\n", prompt) if len(messages) == 2 else None
        if match:
            pid = match[1]
            assert pid in paper_ids and papers[pid] in prompt
            children.append(pid)
            code = f"excerpts = {json.dumps(evidence.get(pid, []))!r}\nprint(FINAL_VAR('excerpts'))"
        else:
            code = next(parent_actions)
        return f"```repl\n{code}\n```"

    prompt = f"{query}\nReturn a JSON list of verbatim passages that answer the question."
    result = LocalRLMRunner(chat, max_depth=2, max_iterations=4).run(json.dumps(papers), prompt)
    expected = [span for pid in paper_ids for span in evidence.get(pid, [])]
    assert result.final_answer is not None
    assert json.loads(result.final_answer) == expected
    assert sorted(children) == sorted(paper_ids)
    assert len(result.child_traces) == len(paper_ids)
    assert sum(message["role"] == "assistant" for message in result.messages) == 4
    assert next(parent_actions, None) is None
    return {"source_turn_ids": [row["id"] for row in rows], "messages": result.messages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace_rows = {row["id"]: row for row in map(json.loads, args.traces.read_text().splitlines())}
    sources = [trace_id.split("-")[1] for trace_id in TRACE_IDS]
    tasks = (
        ds.dataset(args.tasks, format="parquet").to_table(filter=ds.field("source_paper_id").isin(sources)).to_pylist()
    )
    examples = [replay([trace_rows[f"{trace_id}-{i}"] for i in range(1, 5)], tasks) for trace_id in TRACE_IDS]
    args.output.write_text("".join(json.dumps(example) + "\n" for example in examples), encoding="utf-8")


if __name__ == "__main__":
    main()
