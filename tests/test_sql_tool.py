import pytest

from agent.tools import MAX_ROWS, UnsafeQueryError, run_select, validate_select


@pytest.mark.parametrize("query", [
    "SELECT * FROM users",
    "  select count(*) from outcomes;  ",
    "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
    "SELECT created_at_like_column FROM campaigns",  # 'create' kelimesi sütun adında geçebilir
])
def test_select_is_allowed(query):
    validate_select(query)


@pytest.mark.parametrize("query", [
    "DROP TABLE users",
    "DELETE FROM users",
    "UPDATE users SET sehir = 'x'",
    "INSERT INTO users VALUES (1)",
    "SELECT 1; DROP TABLE users",
    "PRAGMA table_info(users)",
    "WITH x AS (DELETE FROM users RETURNING *) SELECT * FROM x",
    "ATTACH DATABASE 'x.db' AS x",
    "SELECT * FROM hidden_truth",
])
def test_non_select_is_rejected(query):
    with pytest.raises(UnsafeQueryError):
        validate_select(query)


def test_row_limit(small_db):
    assert len(run_select("SELECT * FROM users")) == MAX_ROWS


def test_tables_untouched_after_rejected_query(small_db):
    from agent.tools import sql_query
    assert "hata" in sql_query.invoke({"query": "DROP TABLE users"})
    assert run_select("SELECT COUNT(*) AS n FROM users")["n"][0] == 8_000
