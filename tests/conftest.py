import torch


def pytest_sessionstart(session):
    torch.set_num_threads(2)
    torch.manual_seed(1234)
