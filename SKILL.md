---
name: airs-asr
description: Automatic speech recognition.
---

- `airs-asr --help` - Show help.
- `airs-asr --version` - Show version.
- `airs-asr pipe -i <source> -o <target> [-o <target>...] [--backend <name>]` - Transcribe speech. `-i` is required once. `-o` is required at least once and may be repeated. Sources/targets use `file:<path>`, `device`, `device:<name>`, or `stdout`. Defaults: backend=whisper.

Requires sherpa-onnx model files for the selected backend.
