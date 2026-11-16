import pathlib
import uuid

from curation_lab.dedup import ExactDedup


def _tmpdir():
    path = pathlib.Path(__file__).parent / f"_tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_sqlite_exact_dedup_detects_duplicates():
    path = _tmpdir() / "seen.sqlite"
    dedup = ExactDedup(sqlite_path=str(path))
    assert dedup.check_and_add("alpha") is False
    assert dedup.check_and_add("alpha") is True
    assert dedup.check_and_add("beta") is False
    assert dedup.size == 2
    dedup.close()


def test_sqlite_exact_dedup_persists_across_instances():
    path = _tmpdir() / "seen.sqlite"
    first = ExactDedup(sqlite_path=str(path))
    assert first.check_and_add("alpha") is False
    first.close()

    second = ExactDedup(sqlite_path=str(path))
    assert second.check_and_add("alpha") is True
    assert second.check_and_add("beta") is False
    assert second.size == 2
    second.close()


def test_memory_backend_still_works():
    dedup = ExactDedup()
    assert dedup.check_and_add("alpha") is False
    assert dedup.check_and_add("alpha") is True
    assert dedup.size == 1
