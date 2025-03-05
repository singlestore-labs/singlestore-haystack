from typing import Tuple, Dict, Any, Literal


def _convert_filters_to_where_clause_and_params(
        filters: Dict[str, Any], operator: Literal["WHERE", "AND"] = "WHERE"
) -> Tuple[str, Tuple]:
    return "", tuple()
