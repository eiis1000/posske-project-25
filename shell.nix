{ pkgs ? import <nixpkgs> {} }:

let
  my-overrides = pkgs.python313Packages.override {
    overrides = python-self: python-super: {
      # Override the ubelt derivation to disable the test phase.
      ubelt = python-super.ubelt.overrideAttrs (oldAttrs: {
        pytestCheckPhase = "true";
      });
    };
  };
in

pkgs.mkShell {
  name = "sage-with-numba";

  buildInputs = [
    pkgs.sage
    pkgs.python313Packages.numba
    # pkgs.python313Packages.numbaWithCuda
    my-overrides.line_profiler
    pkgs.python313Packages.virtualenv
  ];

  shellHook = ''
    sage --pip install -q numba line_profiler virtualenv
  '';
}
