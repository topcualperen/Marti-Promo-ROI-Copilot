import pytest

from config import get_engine
from data.generate import generate, write_db
from model.uplift import train_and_score


@pytest.fixture(scope="session")
def small_db(tmp_path_factory):
    """Küçük (8.000 kullanıcılı) geçici bir veritabanı kurar ve modeli eğitir."""
    db = tmp_path_factory.mktemp("db") / "test.db"
    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", f"sqlite:///{db.as_posix()}")
    write_db(generate(n_users=8_000, seed=7), get_engine())
    train_and_score(save_plot=False)
    yield
    mp.undo()
