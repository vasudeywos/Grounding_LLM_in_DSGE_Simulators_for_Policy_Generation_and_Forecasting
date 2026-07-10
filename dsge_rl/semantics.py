from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dsge_rl.config import LeverConfig


@dataclass(frozen=True)
class ParsedAction:
    shocks: dict[str, float]
    valid: bool
    raw: str


class ModelSemantics:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.variables, self.shocks = self._load()

    def _load(self) -> tuple[dict[str, str], dict[str, str]]:
        text = self.model_path.read_text()
        variables: dict[str, str] = {}
        shocks: dict[str, str] = {}
        for description, symbol in re.findall(r"#\s+(.+?)\s{2,}([A-Za-z][A-Za-z0-9_]+)(?:\s|$)", text):
            target = shocks if "SHK_" in symbol.upper() or symbol.lower().startswith("e") else variables
            target[symbol] = description.strip()
        try:
            document = yaml.safe_load(text) or {}
            symbols = document.get("symbols", document)
            for key, target in (("variables", variables), ("shocks", shocks)):
                value = symbols.get(key, {}) if isinstance(symbols, dict) else {}
                if isinstance(value, dict):
                    for symbol, description in value.items():
                        target.setdefault(str(symbol), str(description))
                elif isinstance(value, list):
                    for symbol in value:
                        target.setdefault(str(symbol), str(symbol))
        except yaml.YAMLError:
            pass
        return variables, shocks

    def context(self, allowed_shocks: set[str] | None = None) -> str:
        shocks = self.shocks
        if allowed_shocks is not None:
            shocks = {key: value for key, value in shocks.items() if key in allowed_shocks}
        variables = "\n".join(f"{key}: {value}" for key, value in self.variables.items())
        actions = "\n".join(f"{key}: {value}" for key, value in shocks.items())
        return f"Observable variables:\n{variables}\nAvailable policy shocks:\n{actions}".strip()


def parse_action(text: str, levers: tuple[LeverConfig, ...]) -> ParsedAction:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return ParsedAction({}, False, text)
    try:
        data: Any = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ParsedAction({}, False, text)
    if not isinstance(data, dict):
        return ParsedAction({}, False, text)
    lever_name = str(data.get("lever", "")).upper()
    try:
        magnitude = float(data.get("magnitude"))
    except (TypeError, ValueError):
        return ParsedAction({}, False, text)
    lookup = {lever.name.upper(): lever for lever in levers}
    lever = lookup.get(lever_name)
    if lever is None or not lever.minimum <= magnitude <= lever.maximum:
        return ParsedAction({}, False, text)
    return ParsedAction({lever.shock: magnitude}, True, text)

