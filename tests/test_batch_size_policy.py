from config.batch_size_policy import get_batch_size_by_num_clients


def test_batch_size_policy_required_scenarios():
    assert get_batch_size_by_num_clients(10) == 2048
    assert get_batch_size_by_num_clients(20) == 1024
    assert get_batch_size_by_num_clients(50) == 512


def test_batch_size_policy_rejects_unknown_clients():
    try:
        get_batch_size_by_num_clients(7)
    except ValueError:
        return
    raise AssertionError("Expected unknown client count to raise ValueError")
