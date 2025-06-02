#!/bin/bash

# Install uv package manager and set up shell completion
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc

# Install Ruff
uv tool install ruff

# Only run tailscale up if TS_AUTH_KEY is set
if [ -n "$TS_AUTH_KEY" ]; then
  sudo tailscale up --accept-routes --auth-key=$TS_AUTH_KEY
fi