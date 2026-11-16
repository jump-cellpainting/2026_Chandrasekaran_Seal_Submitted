{
  description = "JUMP production pipeline";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            nvidia.acceptLicense = true;
          };
        };

        # Pin pixi to the exact version that writes pixi.lock (the release
        # pixi). nixpkgs-unstable floats pixi (was 0.62.2 when this lock was
        # pinned, now 0.70.1) and a mismatched pixi silently churns pixi.lock
        # (an older one can't read the lock format and regenerates it). Pinning a
        # specific release binary keeps the devShell self-contained AND lock-safe.
        # When bumping pixi, bump this version + hash in lockstep.
        # Pin both workspace platforms. The Linux binary is static musl; the macOS
        # binary is the upstream arm64 release. Other systems fall back to nixpkgs.
        pixi =
          if system == "x86_64-linux" || system == "aarch64-darwin" then
            pkgs.stdenvNoCC.mkDerivation rec {
              pname = "pixi";
              version = "0.68.0";
              src = pkgs.fetchurl {
                url =
                  if system == "x86_64-linux" then
                    "https://github.com/prefix-dev/pixi/releases/download/v${version}/pixi-x86_64-unknown-linux-musl"
                  else
                    "https://github.com/prefix-dev/pixi/releases/download/v${version}/pixi-aarch64-apple-darwin";
                hash =
                  if system == "x86_64-linux" then
                    "sha256-OpRgmobG7Bu2GW8TAw733m3omxQLfhFOX6zWwPoDL8g="
                  else
                    "sha256-yoAEN37AnS6oBohUKea4p+VUxS+WO/PkmSuviLJwbmw=";
              };
              dontUnpack = true;
              installPhase = "install -Dm755 $src $out/bin/pixi";
            }
          else
            pkgs.pixi;

      in
      {
        devShells.default = pkgs.mkShell {
          # Everything the pipeline and `just reproduce` call besides pixi, so a
          # fresh machine needs only Nix. cmake and ninja are for the pixi
          # packages that build from source (rapids-singlecell).
          packages = [ pixi ] ++ (with pkgs; [
            cmake
            direnv
            git
            jq
            just
            ninja
            rclone
            s5cmd
            uv
          ]);

          shellHook = ''
            echo "JUMP Production Pipeline"
            echo ""
            echo "Usage:"
            echo "  pixi install        # Install dependencies"
            echo "  pixi run snakemake  # Run pipeline"
            echo "  just --list         # Show available commands"
          '';

          # Expose the Nix C++ runtime for uv/manylinux wheels and use the
          # system NVIDIA driver libraries to avoid a CUDA version mismatch.
          LD_LIBRARY_PATH =
            pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]
            + ":/run/opengl-driver/lib";
        };

      });
}
