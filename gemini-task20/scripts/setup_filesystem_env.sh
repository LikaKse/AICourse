#!/usr/bin/env bash
# Project-local Node 24 + MCP Filesystem server for gemini-task20.
# Ничего не меняет в системном Node, Homebrew, nvm и общем venv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE_VERSION="24.22.3"
SERVER_VERSION="2.4.0"

case "$(uname -s)" in
  Darwin) os_tag="darwin" ;;
  *) echo "Этот скрипт рассчитан на macOS. Для другой ОС установите Node 24 вручную." >&2; exit 1 ;;
esac

case "$(uname -m)" in
  x86_64|amd64) arch_tag="x64"; sha256="977614af0504c81045e7fd85596dbca0e5f951b96681516dadafedcca8c447f7" ;;
  arm64|aarch64) arch_tag="arm64"; sha256="ff6e17d4874b76d272212b908cc020d6e815fba5a6c7eaaa06add725840307fd" ;;
  *) echo "Неподдерживаемая архитектура: $(uname -m)" >&2; exit 1 ;;
esac

TOOLS_DIR="$ROOT/.tools"
NODE_DIR="$TOOLS_DIR/node-v${NODE_VERSION}-${os_tag}-${arch_tag}"
NODE_URL="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-${os_tag}-${arch_tag}.tar.gz"
SERVER_DIR="$TOOLS_DIR/mcp-server"

mkdir -p "$TOOLS_DIR"

if [ ! -x "$NODE_DIR/bin/node" ]; then
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT
  echo "Скачиваю Node v${NODE_VERSION} (${os_tag}-${arch_tag})..."
  curl -fsSL "$NODE_URL" -o "$tmp_dir/node.tar.gz"
  echo "${sha256}  $tmp_dir/node.tar.gz" | shasum -a 256 -c -
  tar xzf "$tmp_dir/node.tar.gz" -C "$TOOLS_DIR"
  rm -rf "$tmp_dir"
  trap - EXIT
fi

export PATH="$NODE_DIR/bin:$PATH"
NODE="$NODE_DIR/bin/node"
NPM="$NODE_DIR/bin/npm"

actual_node="$("$NODE" -v)"
if [ "$actual_node" != "v${NODE_VERSION}" ]; then
  echo "Ожидался Node v${NODE_VERSION}, получен ${actual_node}" >&2
  exit 1
fi

major="${actual_node#v}"
major="${major%%.*}"
if [ "$major" -lt 24 ]; then
  echo "MCP Filesystem ${SERVER_VERSION} требует Node.js >= 24, найден ${actual_node}" >&2
  exit 1
fi

echo "Node: $actual_node ($NODE)"
echo "npm:  $("$NPM" -v)"

mkdir -p "$SERVER_DIR"
cp "$ROOT/scripts/mcp-server-package.json" "$SERVER_DIR/package.json"
cp "$ROOT/scripts/mcp-server-package-lock.json" "$SERVER_DIR/package-lock.json"

echo "Устанавливаю @modelcontextprotocol/server-filesystem@${SERVER_VERSION}..."
cd "$SERVER_DIR"
"$NPM" ci --ignore-scripts --omit=dev --no-audit --no-fund

installed_version="$("$NODE" -e "const p=require('./node_modules/@modelcontextprotocol/server-filesystem/package.json'); process.stdout.write(p.version)")"
if [ "$installed_version" != "$SERVER_VERSION" ]; then
  echo "Ожидался server-filesystem ${SERVER_VERSION}, установлен ${installed_version}" >&2
  exit 1
fi

echo "MCP Filesystem: ${installed_version}"
echo "Готово. Запускайте: .venv/bin/python main.py --list-tools"
