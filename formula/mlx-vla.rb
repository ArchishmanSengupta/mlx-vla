class MlxVla < Formula
  desc "Vision-Language-Action training framework for Apple Silicon using MLX"
  homepage "https://github.com/ArchishmanSengupta/mlx-vla"
  url "https://github.com/ArchishmanSengupta/mlx-vla/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "REPLACE_WITH_ACTUAL_SHA256"
  license "MIT"

  depends_on "python@3.10"

  def install
    system "pip", "install", "--prefix=#{prefix}", "."
  end

  def caveats
    <<~EOS
      mlx-vla requires Apple Silicon (M1/M2/M3/M4) Mac.

      Usage:
        mlx-vla train --model openvla-7b --dataset bridge_v2
    EOS
  end
end
