use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use airs_audio::AudioFrame;
use async_trait::async_trait;
use futures::{Sink, Stream};
use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineWhisperModelConfig,
};

use crate::backends::{AsrBackend, BackendSessionStream};
use crate::{AsrError, Result};

const WHISPER_LANGUAGE: &str = "en";

/// Whisper speech recognition engine backed by sherpa-onnx.
pub(crate) struct WhisperEngine {
    recognizer: Option<Arc<OfflineRecognizer>>,
    stream: Option<BackendSessionStream>,
}

impl WhisperEngine {
    pub fn new() -> Self {
        Self {
            recognizer: None,
            stream: None,
        }
    }

    /// Search for a model file, accepting both exact names and
    /// sherpa-onnx naming conventions (`{prefix}-encoder.onnx`, etc.).
    fn find_file(dir: &PathBuf, stem: &str) -> Option<PathBuf> {
        // Exact match first
        let path = dir.join(stem);
        if path.exists() {
            return Some(path);
        }
        // Glob-style match: any file ending with `-{stem}` (e.g. `tiny.en-encoder.onnx`)
        let pattern = format!("-{stem}");
        let check = |d: &PathBuf| -> Option<PathBuf> {
            if let Ok(entries) = std::fs::read_dir(d) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    if name.ends_with(&pattern) && name != stem {
                        return Some(entry.path());
                    }
                }
            }
            None
        };
        if let Some(found) = check(dir) {
            return Some(found);
        }
        // Also search one level deep for model subdirectories
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let child = entry.path();
                if child.is_dir() {
                    let nested = child.join(stem);
                    if nested.exists() {
                        return Some(nested);
                    }
                    if let Some(found) = check(&child) {
                        return Some(found);
                    }
                }
            }
        }
        None
    }

    fn to_path_string(path: &PathBuf) -> String {
        path.to_string_lossy().into_owned()
    }

    fn is_english_only_model(path: &PathBuf) -> bool {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();

        name.contains(".en-")
    }

    fn transcribe(recognizer: &OfflineRecognizer, audio: &AudioFrame) -> Result<String> {
        let stream = recognizer.create_stream();
        let samples = crate::backends::mono_samples(audio)?;
        stream.accept_waveform(audio.sample_rate as i32, &samples);
        recognizer.decode(&stream);

        let result = stream
            .get_result()
            .ok_or_else(|| AsrError::Transcription("no recognition result".into()))?;

        Ok(result.text)
    }

    fn ensure_stream(&mut self) -> Result<&mut BackendSessionStream> {
        if self.stream.is_none() {
            let recognizer = self.recognizer.as_ref().ok_or_else(|| {
                AsrError::InvalidInput("Whisper backend not initialized. Call init() first.".into())
            })?;

            self.stream = Some(Box::pin(WhisperStream::new(recognizer.clone())));
        }

        Ok(self.stream.as_mut().expect("ASR stream is initialized"))
    }
}

impl Default for WhisperEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperEngine {
    /// One-shot: transcribe a single audio frame synchronously.
    pub fn process(&self, audio: &AudioFrame) -> Result<String> {
        let recognizer = self.recognizer.as_ref().ok_or_else(|| {
            AsrError::InvalidInput("Whisper backend not initialized. Call init() first.".into())
        })?;
        Self::transcribe(recognizer, audio)
    }
}

#[async_trait]
impl AsrBackend for WhisperEngine {
    async fn init(&mut self) -> Result<()> {
        let model_dir = crate::model_path("whisper");

        let encoder = Self::find_file(&model_dir, "encoder.onnx").ok_or_else(|| {
            AsrError::BackendLoad(format!(
                "encoder.onnx (or *-encoder.onnx) not found in {}",
                model_dir.display()
            ))
        })?;

        let decoder = Self::find_file(&model_dir, "decoder.onnx").ok_or_else(|| {
            AsrError::BackendLoad(format!(
                "decoder.onnx (or *-decoder.onnx) not found in {}",
                model_dir.display()
            ))
        })?;

        let tokens = Self::find_file(&model_dir, "tokens.txt").ok_or_else(|| {
            AsrError::BackendLoad(format!(
                "tokens.txt (or *-tokens.txt) not found in {}",
                model_dir.display()
            ))
        })?;

        if WHISPER_LANGUAGE != "en" && Self::is_english_only_model(&encoder) {
            return Err(AsrError::BackendLoad(format!(
                "configured language is {WHISPER_LANGUAGE}, but the selected Whisper model is English-only: {}. Install a multilingual Whisper model instead of a .en model.",
                encoder.display()
            )));
        }

        let mut config = OfflineRecognizerConfig::default();
        config.model_config = OfflineModelConfig {
            whisper: OfflineWhisperModelConfig {
                encoder: Some(Self::to_path_string(&encoder)),
                decoder: Some(Self::to_path_string(&decoder)),
                language: Some(WHISPER_LANGUAGE.into()),
                task: Some("transcribe".into()),
                tail_paddings: -1,
                enable_token_timestamps: false,
                enable_segment_timestamps: false,
            },
            tokens: Some(Self::to_path_string(&tokens)),
            num_threads: 4,
            provider: Some("cpu".into()),
            ..Default::default()
        };

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| AsrError::BackendLoad("failed to create Whisper recognizer".into()))?;

        self.recognizer = Some(Arc::new(recognizer));
        Ok(())
    }

    async fn process(&mut self, audio: AudioFrame) -> Result<String> {
        WhisperEngine::process(self, &audio)
    }
}

impl Sink<AudioFrame> for WhisperEngine {
    type Error = AsrError;

    fn poll_ready(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Result<()>> {
        match self.ensure_stream() {
            Ok(stream) => stream.as_mut().poll_ready(context),
            Err(error) => Poll::Ready(Err(error)),
        }
    }

    fn start_send(mut self: Pin<&mut Self>, item: AudioFrame) -> Result<()> {
        self.ensure_stream()?.as_mut().start_send(item)
    }

    fn poll_flush(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Result<()>> {
        match self.ensure_stream() {
            Ok(stream) => stream.as_mut().poll_flush(context),
            Err(error) => Poll::Ready(Err(error)),
        }
    }

    fn poll_close(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Result<()>> {
        match self.ensure_stream() {
            Ok(stream) => stream.as_mut().poll_close(context),
            Err(error) => Poll::Ready(Err(error)),
        }
    }
}

impl Stream for WhisperEngine {
    type Item = Result<String>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.ensure_stream() {
            Ok(stream) => stream.as_mut().poll_next(context),
            Err(error) => Poll::Ready(Some(Err(error))),
        }
    }
}

struct WhisperStream {
    recognizer: Arc<OfflineRecognizer>,
    samples: Vec<f32>,
    channels: Option<u16>,
    sample_rate: Option<u32>,
    pending: std::collections::VecDeque<Result<String>>,
    closed: bool,
    waker: Option<Waker>,
}

impl WhisperStream {
    fn new(recognizer: Arc<OfflineRecognizer>) -> Self {
        Self {
            recognizer,
            samples: Vec::new(),
            channels: None,
            sample_rate: None,
            pending: std::collections::VecDeque::new(),
            closed: false,
            waker: None,
        }
    }

    fn push_result(&mut self, result: Result<String>) {
        self.pending.push_back(result);
        if let Some(waker) = self.waker.take() {
            waker.wake();
        }
    }
}

impl Sink<AudioFrame> for WhisperStream {
    type Error = AsrError;

    fn poll_ready(self: std::pin::Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(mut self: std::pin::Pin<&mut Self>, item: AudioFrame) -> Result<()> {
        if self.closed {
            return Err(AsrError::InvalidInput(
                "cannot send audio after ASR stream is closed".into(),
            ));
        }

        if let Some(channels) = self.channels {
            if channels != item.channels {
                return Err(AsrError::InvalidInput(
                    "audio channel count changed during ASR stream".into(),
                ));
            }
        } else {
            self.channels = Some(item.channels);
        }

        if let Some(sample_rate) = self.sample_rate {
            if sample_rate != item.sample_rate {
                return Err(AsrError::InvalidInput(
                    "audio sample rate changed during ASR stream".into(),
                ));
            }
        } else {
            self.sample_rate = Some(item.sample_rate);
        }

        let samples = crate::backends::mono_samples(&item)?;
        self.samples.extend_from_slice(&samples);
        Ok(())
    }

    fn poll_flush(self: std::pin::Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(
        mut self: std::pin::Pin<&mut Self>,
        _context: &mut Context<'_>,
    ) -> Poll<Result<()>> {
        if !self.closed {
            self.closed = true;

            let audio = AudioFrame {
                samples: std::mem::take(&mut self.samples),
                channels: 1,
                sample_rate: self.sample_rate.unwrap_or(16000),
            };

            let result = Self::recognize(&self.recognizer, &audio);
            self.push_result(result);
        }

        Poll::Ready(Ok(()))
    }
}

impl WhisperStream {
    fn recognize(recognizer: &OfflineRecognizer, audio: &AudioFrame) -> Result<String> {
        WhisperEngine::transcribe(recognizer, audio)
    }
}

impl Stream for WhisperStream {
    type Item = Result<String>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if let Some(result) = self.pending.pop_front() {
            return Poll::Ready(Some(result));
        }

        if self.closed {
            return Poll::Ready(None);
        }

        self.waker = Some(context.waker().clone());
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_samples_keeps_mono_audio() {
        let audio = AudioFrame {
            samples: vec![0.25, -0.5],
            channels: 1,
            sample_rate: 16000,
        };

        assert_eq!(
            crate::backends::mono_samples(&audio).unwrap(),
            audio.samples
        );
    }

    #[test]
    fn mono_samples_downmixes_stereo_audio() {
        let audio = AudioFrame {
            samples: vec![1.0, -1.0, 0.5, 0.25],
            channels: 2,
            sample_rate: 16000,
        };

        assert_eq!(
            crate::backends::mono_samples(&audio).unwrap(),
            vec![0.0, 0.375]
        );
    }

    #[test]
    fn mono_samples_rejects_misaligned_audio() {
        let audio = AudioFrame {
            samples: vec![1.0, 0.0, -1.0],
            channels: 2,
            sample_rate: 16000,
        };

        assert!(matches!(
            crate::backends::mono_samples(&audio),
            Err(AsrError::InvalidInput(_))
        ));
    }

    #[test]
    fn english_only_model_detection_uses_filename() {
        assert!(WhisperEngine::is_english_only_model(&PathBuf::from(
            "tiny.en-encoder.onnx"
        )));
        assert!(!WhisperEngine::is_english_only_model(&PathBuf::from(
            "tiny-encoder.onnx"
        )));
    }
}
