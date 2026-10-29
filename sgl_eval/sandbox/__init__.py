"""Execution backends for agentic benchmarks.

Container lifecycle, file transfer and the verifier come from the vendored
pier ``DockerEnvironment``; this package adds only what the host-side agent
loop needs on top of it (per-action command execution and host preflight).
"""
