from typing import TypedDict


class RequiredChampionResult(TypedDict):
    name: str
    champ1: str
    champ2: str


class ChampionResult(RequiredChampionResult, total=False):
    p_value: float
    cohen_d: float
