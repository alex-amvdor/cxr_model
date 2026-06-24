# Create Regression Tests

Create regression tests for: $ARGUMENTS

Requirements:

1. Preserve numerical behavior.
2. Use deterministic RNG seeds.
3. Use physically meaningful test cases.
4. Verify:
   - shapes
   - units
   - normalization
   - numerical tolerances
5. Prefer analytic comparisons when possible.
6. Keep tests fast.

Report:

- new tests
- expected tolerances
- rationale
