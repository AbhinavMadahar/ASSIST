# ASSIST AI

This is the AI powering ASSIST. It is based on a neural network.

To use it, run
```sh
./bin/prognosis
```
and fill out the fields it requests by passing them into STDIN.

Do **NOT** interact with the AI by directly calling a file in `src/`. These
files may change name or structure, which can break other parts of the code.
Instead, always only use files in `bin/`, which are much more stable.
