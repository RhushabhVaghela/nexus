from nexus.core.checkpointing.activation_checkpointer import ActivationCheckpointer


def test_checkpointer_saves_and_retrieves():
    cp = ActivationCheckpointer(checkpoint_frequency=2)
    cp.maybe_save(0, "a0")
    cp.maybe_save(1, "a1")
    cp.maybe_save(2, "a2")

    assert cp.get(0) == "a0"
    assert cp.get(1) is None
    assert cp.get(2) == "a2"

    cp.clear()
    assert cp.get(0) is None
