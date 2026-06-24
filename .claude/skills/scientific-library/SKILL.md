# Scientific library conventions

- All simulation kernels live in src/cxr_model/.
- Avoid introducing new dependencies.
- Prefer NumPy/CuPy vectorization.
- Avoid OOP unless stateful behavior is required.
- Public APIs require docstrings and type hints.
- Never duplicate physics constants.
