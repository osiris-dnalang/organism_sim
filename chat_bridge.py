#!/usr/bin/env python3
"""Neuro-symbolic bridge REPL: English ↔ dna::}{::lang genes ↔ organism_sim runtime.

    python chat_bridge.py --model qwen2.5:7b --warmup 500
    python chat_bridge.py --help

Implementation: ``organism_sim.chat_bridge``. Needs a local Ollama; the substrate itself
never calls a model.
"""
import sys

from organism_sim.chat_bridge import main

if __name__ == "__main__":
    sys.exit(main())
