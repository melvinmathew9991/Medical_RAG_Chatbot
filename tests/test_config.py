"""
Checks on `medbot.config`'s environment handling.

Runnable with pytest, or standalone:

    .venv-gemini/Scripts/python.exe -m tests.test_config

The thing actually worth pinning here is the `or` fallback documented at
config.py:16-19. `.env` declares DATA_DIR/PERSIST_DIR/GEMINI_CHAT_MODEL as blank
lines for discoverability; python-dotenv turns a blank line into "", and
`os.getenv("X", default)` only returns `default` when the variable is fully
*absent*. So the obvious-looking `os.getenv("DATA_DIR", fallback)` would hand
back "" and the app would look for the corpus in the current directory. The
comment says so; these tests make it fail loudly if someone "simplifies" it.
"""

import contextlib
import importlib
import os
from unittest import mock

import medbot.config


@contextlib.contextmanager
def config_with_env(**env):
    """
    Re-import `medbot.config` under a controlled environment.

    Two things make this less direct than it looks. `config` reads os.environ at
    import time, so reloading is the only way to observe different inputs. And
    `load_dotenv` is stubbed out for the duration, because otherwise a developer's
    real `.env` would leak into the result and these tests would pass or fail
    depending on whose machine ran them.

    The reload in the `finally` runs unpatched, restoring the genuine module state
    for anything that imports config later in the session.
    """
    try:
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("dotenv.load_dotenv"):
            yield importlib.reload(medbot.config)
    finally:
        importlib.reload(medbot.config)


def test_blank_env_var_falls_back_to_default():
    """The bug the `or` guards against: dotenv writes "" and getenv's default never fires."""
    with config_with_env(DATA_DIR="", PERSIST_DIR="", GEMINI_CHAT_MODEL="") as cfg:
        assert cfg.DATA_DIR == os.path.join(cfg.BASE_DIR, "data")
        assert cfg.PERSIST_DIR == os.path.join(cfg.BASE_DIR, "vectorstore")
        assert cfg.GEMINI_CHAT_MODEL == "gemini-flash-lite-latest"


def test_absent_env_var_falls_back_to_default():
    with config_with_env() as cfg:
        assert cfg.DATA_DIR == os.path.join(cfg.BASE_DIR, "data")
        assert cfg.PERSIST_DIR == os.path.join(cfg.BASE_DIR, "vectorstore")
        assert cfg.GEMINI_CHAT_MODEL == "gemini-flash-lite-latest"
        assert cfg.LOCAL_EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"


def test_set_env_var_wins_over_default():
    """The fallback must not be so eager that it ignores a real override."""
    with config_with_env(
        DATA_DIR="/tmp/corpus",
        PERSIST_DIR="/tmp/index",
        GEMINI_CHAT_MODEL="gemini-something-else",
        LOCAL_EMBEDDING_MODEL="BAAI/bge-large-en-v1.5",
    ) as cfg:
        assert cfg.DATA_DIR == "/tmp/corpus"
        assert cfg.PERSIST_DIR == "/tmp/index"
        assert cfg.GEMINI_CHAT_MODEL == "gemini-something-else"
        assert cfg.LOCAL_EMBEDDING_MODEL == "BAAI/bge-large-en-v1.5"


def test_base_dir_is_the_repo_root():
    """DATA_DIR/PERSIST_DIR defaults are only correct if BASE_DIR is the repo root."""
    with config_with_env() as cfg:
        assert os.path.isdir(os.path.join(cfg.BASE_DIR, "medbot"))
        assert os.path.isfile(os.path.join(cfg.BASE_DIR, "requirements.txt"))


def test_langsmith_tracing_stays_off_without_a_key():
    """Enabling tracing with no key makes every call emit a failing background request."""
    with config_with_env() as cfg:
        assert cfg.LANGCHAIN_API_KEY == ""
        assert "LANGCHAIN_TRACING_V2" not in os.environ


def test_langsmith_tracing_turns_on_with_a_key():
    with config_with_env(LANGCHAIN_API_KEY="ls-fake") as cfg:
        assert cfg.LANGCHAIN_API_KEY == "ls-fake"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"] == "ls-fake"


def test_blank_langsmith_key_does_not_turn_tracing_on():
    """`.env` ships this blank too, so "" must be treated as absent here as well."""
    with config_with_env(LANGCHAIN_API_KEY=""):
        assert "LANGCHAIN_TRACING_V2" not in os.environ


def test_google_api_key_is_republished_to_the_environment():
    """The Gemini SDK reads the env var directly, not the config attribute."""
    with config_with_env(GOOGLE_API_KEY="AIza-fake") as cfg:
        assert cfg.GOOGLE_API_KEY == "AIza-fake"
        assert os.environ["GOOGLE_API_KEY"] == "AIza-fake"


def test_missing_google_api_key_is_empty_not_none():
    """Downstream code does truthiness checks; None vs "" would still work, but
    SERPAPI_API_KEY is deliberately None-able and these two must not be confused."""
    with config_with_env() as cfg:
        assert cfg.GOOGLE_API_KEY == ""
        assert cfg.SERPAPI_API_KEY is None


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
