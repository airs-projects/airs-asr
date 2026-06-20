# airs-asr

This document lists the CLI and public API.

## CLI

- `airs-asr --help` - Show help.
- `airs-asr --version` - Show version.
- `airs-asr pipe -i <source> -o <target> [-o <target>...] [--backend <name>]` - Transcribe speech.

`-i` is required once. `-o` is required at least once and may be repeated.
Backends: `whisper`, `zipformer`.

Sources:
- `file:<path>`
- `device`
- `device:<name>`

Targets:
- `file:<path>`
- `stdout`

```
airs-asr pipe -i file:speech.wav -o stdout
airs-asr pipe -i device -o file:transcript.txt
airs-asr pipe -i device:Yeti -o file:out.txt -o stdout
```

## Library public API

- `version()` - Return the crate version string.
- `Result<T>` - Library result type using `AsrError`.
- `AsrError` - Error enum for invalid input, backend load, transcription, and audio failures.

- `Processor` - Speech-to-text processor with chainable backend configuration. Implements `Sink<AudioFrame, Error = AsrError>` and `Stream<Item = Result<String>>`.
- `Processor::new()` - Create a new processor with default backend.
- `AsrBackendKind` - Backend selection enum (e.g. `AsrBackendKind::Whisper`, `AsrBackendKind::Zipformer`).
- `Processor::set_backend(kind)` - Set the backend implementation.
- `Processor::init()` - Async; load the selected backend before transcription.
- `Processor::process(audio)` - Async; one-shot transcription: feed an audio frame and return recognized text.
- `Processor::is_ready()` - Return whether the selected backend has been initialized.
- `AsrBackend` - Backend trait implemented by streaming ASR backends; backends implement `Sink<AudioFrame>` and `Stream<Result<String>>`.
