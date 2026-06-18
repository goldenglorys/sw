"""Environment (EN) module: agent-operable, self-resetting, self-verifying tasks."""

from .base import BaseEnv, StepResult
from .pusht import PushTEnv, GymPushTEnv


def make_env(name: str = "pusht", backend: str = "builtin", **kwargs) -> BaseEnv:
    """Factory returning an environment instance by name and backend.

    Args:
        name: Task name. Currently ``"pusht"`` is supported.
        backend: ``"builtin"`` for the numpy-only simplified physics, or
            ``"gym_pusht"`` for the high-fidelity official environment.
        **kwargs: Forwarded to the environment constructor.

    Returns:
        A :class:`BaseEnv` instance.

    Raises:
        ValueError: If the task name is unknown.
    """
    name = name.lower()
    if name in ("pusht", "push-t", "push_t"):
        if backend == "gym_pusht":
            return GymPushTEnv(**kwargs)
        return PushTEnv(**kwargs)
    raise ValueError(f"Unknown environment '{name}'. Available: ['pusht'].")


__all__ = ["BaseEnv", "StepResult", "PushTEnv", "GymPushTEnv", "make_env"]
