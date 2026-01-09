# SPDX-FileCopyrightText: 2025-present SingleStore, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any, Literal

from haystack.errors import FilterError


def _convert_filters_to_where_clause_and_params(
    filters: dict[str, Any], operator: Literal["WHERE", "AND"] = "WHERE"
) -> tuple[str, tuple]:
    """
    Convert Haystack filters to a WHERE clause and a tuple of params to query SingleStore.
    """
    if "field" in filters:
        query, params = _parse_comparison_condition(filters)
    else:
        query, params = _parse_logical_condition(filters)

    where_clause = f" {operator} {query}"

    return where_clause, tuple(params)


def _parse_logical_condition(condition: dict[str, Any]) -> tuple[str, list[Any]]:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    if operator not in ["AND", "OR", "NOT"]:
        msg = f"Unknown logical operator '{operator}'. Valid operators are: 'AND', 'OR'"
        raise FilterError(msg)

    # logical conditions can be nested, so we need to parse them recursively
    query_parts, values = [], []
    for c in condition["conditions"]:
        if "field" in c:
            query, vals = _parse_comparison_condition(c)
        else:
            query, vals = _parse_logical_condition(c)
        query_parts.append(query)
        values.extend(vals)

    if operator == "AND":
        sql_query = f"({' AND '.join(query_parts)})"
    elif operator == "OR":
        sql_query = f"({' OR '.join(query_parts)})"
    elif operator == "NOT":
        sql_query = f"(NOT ({' AND '.join(query_parts)})) OR ({' AND '.join(query_parts)}) IS NULL"
        values.extend(values)
    else:
        msg = f"Unknown logical operator '{operator}'"
        raise FilterError(msg)

    return sql_query, values


def _parse_comparison_condition(condition: dict[str, Any]) -> tuple[str, list[Any]]:
    field: str = condition["field"]
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'. Valid operators are: {list(COMPARISON_OPERATORS.keys())}"
        raise FilterError(msg)

    value: Any = condition["value"]

    if field.startswith("meta."):
        field = _treat_meta_field(field, value)

    field, value = COMPARISON_OPERATORS[operator](field, value)
    return field, value


def _treat_meta_field(field: str, value: Any) -> str:
    """
    Internal method that modifies the field str
    to make the meta JSON field queryable.
    """

    path = field.split(".")
    final_field = path[-1]
    prefix = "::".join(path[:-1])

    if isinstance(value, str):
        return f"{prefix}::${final_field}"
    else:
        return f"{prefix}::{final_field}"


def _equal(field: str, value: Any) -> tuple[str, list[Any]]:
    if value is None:
        return f"{field} IS NULL", []
    return f"{field} = %s", [value]


def _not_equal(field: str, value: Any) -> tuple[str, list[Any]]:
    if value is None:
        return f"{field} IS NOT NULL", []
    return f"{field} IS NULL OR {field} != %s", [value]


def _greater_than(field: str, value: Any) -> tuple[str, list[Any]]:
    _comparable_value_check(value)
    return f"{field} > %s", [value]


def _greater_than_equal(field: str, value: Any) -> tuple[str, list[Any]]:
    _comparable_value_check(value)
    return f"{field} >= %s", [value]


def _less_than(field: str, value: Any) -> tuple[str, list[Any]]:
    _comparable_value_check(value)
    return f"{field} < %s", [value]


def _less_than_equal(field: str, value: Any) -> tuple[str, list[Any]]:
    _comparable_value_check(value)
    return f"{field} <= %s", [value]


def _comparable_value_check(value: Any) -> None:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if isinstance(value, list):
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)


def _not_in(field: str, value: Any) -> tuple[str, list[Any]]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'not in' comparator"
        raise FilterError(msg)

    return f"{field} IS NULL OR {field} NOT IN({','.join(['%s'] * len(value))})", value


def _in(field: str, value: Any) -> tuple[str, list[Any]]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' comparator"
        raise FilterError(msg)

    return f"{field} IN({','.join(['%s'] * len(value))})", value


def _like(field: str, value: Any) -> tuple[str, list[Any]]:
    if not isinstance(value, str):
        msg = f"{field}'s value must be a str when using 'LIKE' "
        raise FilterError(msg)
    return f"{field} LIKE %s", [value]


def _not_like(field: str, value: Any) -> tuple[str, list[Any]]:
    if not isinstance(value, str):
        msg = f"{field}'s value must be a str when using 'NOT LIKE' "
        raise FilterError(msg)
    return f"{field} NOT LIKE %s", [value]


COMPARISON_OPERATORS = {
    "==": _equal,
    "!=": _not_equal,
    ">": _greater_than,
    ">=": _greater_than_equal,
    "<": _less_than,
    "<=": _less_than_equal,
    "in": _in,
    "not in": _not_in,
    "like": _like,
    "not like": _not_like,
}
