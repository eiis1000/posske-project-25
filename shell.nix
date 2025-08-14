{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "sage-with-numba";

  buildInputs = [
    pkgs.sage
    pkgs.python3Packages.numba
  ];

  shellHook = ''
    sage --pip install -q numba
  '';
}
