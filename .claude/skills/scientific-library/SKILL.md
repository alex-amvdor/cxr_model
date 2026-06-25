# Scientific library conventions

- All simulation kernels live in src/cxr_mc/.
- Prefer NumPy/CuPy vectorization.
- Avoid OOP unless stateful behavior is required.
- Public APIs require docstrings and type hints.
- Never duplicate physics constants.
- Prefer scipy.constants or similar to hard-coded values
