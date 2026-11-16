"""Сложность стратегий — по фактическому коду (ast + tokenize), без субъективных баллов.

Для каждой стратегии перечислены ЕЁ компоненты (файлы, классы, функции). Общий для всех код
(strategies/base.py, Agent.ask, _generate, подсчёт токенов, хранение полной истории) не учитывается.

Метрики:
* loc — строки кода без пустых, комментариев и docstring;
* components — классы + функции верхнего уровня в компонентах стратегии;
* functions — все def (включая методы);
* extra_state_files — файлы состояния сверх full_history.json;
* extra_model_call_types — виды служебных вызовов Gemini (summary / извлечение facts);
* extra_prompts — дополнительные системные инструкции в config.py;
* model_output_validation — проверяется ли ответ модели кодом перед применением.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import config

COMPONENTS: dict[str, list[tuple[str, str | None]]] = {
    "sliding": [("strategies/sliding_window.py", None)],
    "facts": [("strategies/sticky_facts.py", None), ("facts_store.py", None),
              ("agent.py", "build_facts_request"), ("agent.py", "Agent.extract_facts")],
    "summary": [("strategies/summary.py", None),
                ("context_manager.py", "SummarizationError"), ("context_manager.py", "StateError"),
                ("context_manager.py", "SummaryOutput"), ("context_manager.py", "CompressionEvent"),
                ("context_manager.py", "ContextManager"),
                ("history_store.py", "SummaryState"), ("history_store.py", "SummaryStore"),
                ("agent.py", "render_messages"), ("agent.py", "build_summary_request"), ("agent.py", "Agent.summarize")],
}
EXTRA = {
    "sliding": {"extra_model_call_types": 0, "extra_prompts": 0, "model_output_validation": False},
    "facts": {"extra_model_call_types": 1, "extra_prompts": 2, "model_output_validation": True},
    "summary": {"extra_model_call_types": 1, "extra_prompts": 1, "model_output_validation": False},
}


def _code_lines(source: str, tree: ast.AST) -> set[int]:
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                doc_lines |= set(range(first.lineno, first.end_lineno + 1))
    lines: set[int] = set()
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type not in skip:
            lines |= set(range(tok.start[0], tok.end[0] + 1))
    return lines - doc_lines


def _find(tree: ast.Module, qualname: str) -> ast.AST:
    node: ast.AST = tree
    for part in qualname.split("."):
        node = next(n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part)
    return node


def measure(strategy: str, root: Path = config.ROOT) -> dict:
    loc = functions = components = 0
    units = []
    for rel, qual in COMPONENTS[strategy]:
        source = (root / rel).read_text(encoding="utf-8")
        tree = ast.parse(source)
        code = _code_lines(source, tree)
        if qual is None:
            nodes = [tree]
            components += sum(isinstance(n, (ast.ClassDef, ast.FunctionDef)) for n in tree.body)
            span = code
        else:
            node = _find(tree, qual)
            nodes = [node]
            components += 1 if "." not in qual else 0      # метод — часть уже посчитанного класса Agent
            span = {l for l in code if node.lineno <= l <= node.end_lineno}
        loc += len(span)
        functions += sum(isinstance(n, ast.FunctionDef) for x in nodes for n in ast.walk(x))
        units.append(f"{rel}" + (f"::{qual}" if qual else ""))
    from strategies import strategy_class
    return {"loc": loc, "components": components, "functions": functions,
            "extra_state_files": len(strategy_class(strategy).state_files) - 1,
            **EXTRA[strategy], "units": units}


def measure_all() -> dict:
    return {s: measure(s) for s in config.STRATEGIES}


if __name__ == "__main__":
    for name, m in measure_all().items():
        print(name, {k: v for k, v in m.items() if k != "units"})
