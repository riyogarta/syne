"""Tests for SQL statement splitting in syne/cli/helpers.py."""

from syne.cli.helpers import _split_sql_statements


class TestSplitSqlStatements:
    def test_single_create_table(self):
        sql = "CREATE TABLE t (id INT);"
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 1
        assert "CREATE TABLE" in stmts[0]

    def test_two_statements(self):
        sql = "CREATE TABLE a (id INT);\nCREATE TABLE b (id INT);"
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 2

    def test_do_block(self):
        sql = """DO $$
BEGIN
    EXECUTE 'ALTER TABLE t ADD COLUMN x INT';
END $$;"""
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 1
        assert "DO $$" in stmts[0]
        assert "END $$;" in stmts[0]

    def test_create_function(self):
        sql = """CREATE OR REPLACE FUNCTION foo()
RETURNS void AS $$
BEGIN
    RAISE NOTICE 'hello';
END;
$$ LANGUAGE plpgsql;"""
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 1
        assert "CREATE OR REPLACE FUNCTION" in stmts[0]

    def test_mixed_create_and_do(self):
        sql = """CREATE TABLE t (id INT);

DO $$
BEGIN
    EXECUTE 'SELECT 1';
END $$;

CREATE TABLE t2 (id INT);"""
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 3

    def test_comments_between_statements(self):
        sql = """-- Comment at top
CREATE TABLE a (id INT);
-- Another comment
CREATE TABLE b (id INT);"""
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 2

    def test_comment_inside_statement(self):
        sql = """CREATE TABLE t (
    -- a column
    id INT,
    name TEXT
);"""
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 1
        assert "-- a column" in stmts[0]

    def test_empty_input(self):
        assert _split_sql_statements("") == []

    def test_only_comments(self):
        sql = "-- just comments\n-- nothing else"
        assert _split_sql_statements(sql) == []

    def test_trailing_without_semicolon(self):
        sql = "SELECT 1"
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 1
        assert stmts[0].strip() == "SELECT 1"

    def test_multiline_insert(self):
        sql = """INSERT INTO config (key, value) VALUES
    ('a', '"b"'),
    ('c', '"d"')
ON CONFLICT (key) DO NOTHING;"""
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 1
        assert "ON CONFLICT" in stmts[0]

    def test_create_index_multiline(self):
        sql = """CREATE UNIQUE INDEX IF NOT EXISTS idx_name
    ON my_table (LOWER(name), type);"""
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 1

    def test_real_schema_fragment(self):
        """Test with a fragment similar to the actual schema.sql."""
        sql = """CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memory (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector
);

CREATE INDEX IF NOT EXISTS idx_memory ON memory (id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory' AND column_name = 'permanent'
    ) THEN
        ALTER TABLE memory ADD COLUMN permanent BOOLEAN DEFAULT false;
    END IF;
END $$;

INSERT INTO config (key, value) VALUES
    ('graph.enabled', 'true')
ON CONFLICT (key) DO NOTHING;"""

        stmts = _split_sql_statements(sql)
        assert len(stmts) == 5
        assert "CREATE EXTENSION" in stmts[0]
        assert "CREATE TABLE" in stmts[1]
        assert "CREATE INDEX" in stmts[2]
        assert "DO $$" in stmts[3]
        assert "INSERT INTO" in stmts[4]
