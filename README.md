# Integrable Deformations of Spin Chains
## Summary
This code implements the spin chain deformation procedure described in ["Long-Range Deformations for Integrable Spin Chains"](https://arxiv.org/abs/0902.0956) by Bargheer et. al.
(J.Phys.A42:285205,2009) but with a few differences in its operator definitions/identifications. These differences are described in the documentation for the operators in question. This project was completed under the supervision of Thore Posske and Sergio Hörtner of the [Posske Research Group for condensed matter & topology](posske.de) at the University of Hamburg.


## Usage
Install `nix-shell`, run `nix-shell` to launch the shell in the appropriate environment, then run the code via e.g. `sage --python run_tests.py` (or in one line with `nix-shell --run "sage --python run_tests.py"`). Using any environment with both `sage` and `numba` installed in the same Python environment (>=3.13) would also work, but using `nix-shell` makes it particularly easy to install the dependencies correctly. The code in `deformations/scratch.py` demonstrates some usage examples and can be run via `sage --python run-scratch.py`. The code can also be profiled via e.g. `sage --python -m kernprof -lv run_tests.py` in the `nix` shell.


## To-Do Items
- [ ] Document methods
- [ ] Figure out why algebra refactor is slow (c.f. `no-algebra-refactor` branch)
- [ ] Write a proper README.md
- [ ] Figure out why the boost-bilocal conversion square doesn't commute (see tests.py)
- [ ] Switch logging to use `logging` package
