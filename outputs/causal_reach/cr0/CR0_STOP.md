# CR-0 STOP

Content-state availability failed on the frozen grid.

- content_gate.pass = False
- content_gate.reasons = ['n_pairs 9 < 12']
- n_content_pairs = 9
- mode_gate.pass = True (cannot substitute)
- n_mode_pairs = 21

Do not start CR-1. Do not scan layers. Do not fit U. Do not patch. Do not change ε, steps, temperature, or the refusal-margin objective to manufacture pairs.
Stop the content-path CausalReach line.
