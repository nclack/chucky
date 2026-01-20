{
  description = "Development environment for chucky";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self, # required even if the lsp complains
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        formatter = pkgs.nixfmt-tree;

        devShells.default = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
          name = "chucky";

          buildInputs = with pkgs; [
            # clang-tools
            cmake
            cudatoolkit
            gh
            # entr
            # lldb
            man-pages
            man-pages-posix
            ninja
            nixd
            pkg-config
            spdlog
          ];

          # CPATH = "${pkgs.llvmPackages.libcxx.dev}/include/c++/v1";
          # CPLUS_INCLUDE_PATH = "${pkgs.llvmPackages.libcxx.dev}/include/c++/v1";

          # LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          # NVCOMP_LIB = "${pkgs.cudaPackages.nvcomp.static}";
          # NVCOMP_INCLUDE = "${pkgs.cudaPackages.nvcomp.include}/include";
          # CUDA_NVCC_PATH = "${pkgs.cudaPackages.cuda_nvcc}";

          # CUDA_RUNTIME_INCLUDE = "${pkgs.cudatoolkit}/include";
          # CUDA_RUNTIME_LIB = "${pkgs.cudatoolkit}/lib";
        };
      }
    );
}
