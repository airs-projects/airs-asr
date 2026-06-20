#[cfg(any(feature = "whisper", feature = "zipformer"))]
use std::path::PathBuf;
use std::pin::Pin;
use std::task::{Context, Poll};

use airs_audio::AudioFrame;
use futures::{Sink, Stream};

mod backends;

pub use backends::AsrBackend;

#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("failed to load ASR backend: {0}")]
    BackendLoad(String),
    #[error("ASR transcription failed: {0}")]
    Transcription(String),
    #[error("audio error: {0}")]
    Audio(#[from] airs_audio::AudioError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, AsrError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsrBackendKind {
    Whisper,
    Zipformer,
}

impl Default for AsrBackendKind {
    fn default() -> Self {
        Self::Whisper
    }
}

/// Speech-to-text processor with chainable backend configuration.
pub struct Processor {
    backend_kind: AsrBackendKind,
    backend: Option<Box<dyn AsrBackend>>,
    is_ready: bool,
}

impl Default for Processor {
    fn default() -> Self {
        Self {
            backend_kind: AsrBackendKind::default(),
            backend: None,
            is_ready: false,
        }
    }
}

impl Processor {
    /// Create a new ASR processor with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the backend implementation.
    pub fn set_backend(mut self, kind: AsrBackendKind) -> Self {
        self.backend_kind = kind;
        self.backend = None;
        self.is_ready = false;
        self
    }

    /// Load the selected backend before the first transcription request.
    pub async fn init(mut self) -> Result<Self> {
        if self.backend.is_none() {
            self.backend = Some(create_backend(self.backend_kind)?);
        }

        let backend = self.backend.as_mut().unwrap();
        backend.init().await?;
        self.is_ready = true;
        Ok(self)
    }

    /// Return whether the selected backend has been initialized.
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }

    /// One-shot: transcribe a single audio frame and return the recognized text.
    pub async fn process(&mut self, audio: AudioFrame) -> Result<String> {
        if !self.is_ready {
            return Err(AsrError::InvalidInput(
                "ASR backend is not initialized. Call init() first.".to_string(),
            ));
        }

        self.backend.as_mut().unwrap().process(audio).await
    }
}

impl Sink<AudioFrame> for Processor {
    type Error = AsrError;

    fn poll_ready(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Result<()>> {
        if !self.is_ready {
            return Poll::Ready(Err(AsrError::InvalidInput(
                "ASR backend is not initialized. Call init() first.".to_string(),
            )));
        }

        let backend = self.backend.as_mut().expect("ready engine has backend");
        Pin::new(&mut **backend).poll_ready(context)
    }

    fn start_send(mut self: Pin<&mut Self>, item: AudioFrame) -> Result<()> {
        if !self.is_ready {
            return Err(AsrError::InvalidInput(
                "ASR backend is not initialized. Call init() first.".to_string(),
            ));
        }

        let backend = self.backend.as_mut().expect("ready engine has backend");
        Pin::new(&mut **backend).start_send(item)
    }

    fn poll_flush(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Result<()>> {
        if !self.is_ready {
            return Poll::Ready(Err(AsrError::InvalidInput(
                "ASR backend is not initialized. Call init() first.".to_string(),
            )));
        }

        let backend = self.backend.as_mut().expect("ready engine has backend");
        Pin::new(&mut **backend).poll_flush(context)
    }

    fn poll_close(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Result<()>> {
        if !self.is_ready {
            return Poll::Ready(Err(AsrError::InvalidInput(
                "ASR backend is not initialized. Call init() first.".to_string(),
            )));
        }

        let backend = self.backend.as_mut().expect("ready engine has backend");
        Pin::new(&mut **backend).poll_close(context)
    }
}

impl Stream for Processor {
    type Item = Result<String>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if !self.is_ready {
            return Poll::Ready(Some(Err(AsrError::InvalidInput(
                "ASR backend is not initialized. Call init() first.".to_string(),
            ))));
        }

        let backend = self.backend.as_mut().expect("ready engine has backend");
        Pin::new(&mut **backend).poll_next(context)
    }
}

fn create_backend(kind: AsrBackendKind) -> Result<Box<dyn AsrBackend>> {
    match kind {
        AsrBackendKind::Whisper => create_whisper_backend(),
        AsrBackendKind::Zipformer => create_zipformer_backend(),
    }
}

#[cfg(any(not(feature = "whisper"), not(feature = "zipformer")))]
fn backend_feature_disabled(name: &str) -> Result<Box<dyn AsrBackend>> {
    Err(AsrError::InvalidInput(format!(
        "{name} ASR backend feature is not enabled"
    )))
}

#[cfg(feature = "whisper")]
fn create_whisper_backend() -> Result<Box<dyn AsrBackend>> {
    Ok(Box::new(backends::whisper::WhisperEngine::new()))
}

#[cfg(not(feature = "whisper"))]
fn create_whisper_backend() -> Result<Box<dyn AsrBackend>> {
    backend_feature_disabled("whisper")
}

#[cfg(feature = "zipformer")]
fn create_zipformer_backend() -> Result<Box<dyn AsrBackend>> {
    Ok(Box::new(backends::zipformer::ZipformerEngine::new()))
}

#[cfg(not(feature = "zipformer"))]
fn create_zipformer_backend() -> Result<Box<dyn AsrBackend>> {
    backend_feature_disabled("zipformer")
}

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(any(feature = "whisper", feature = "zipformer"))]
fn model_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    PathBuf::from(home).join(".airs/models")
}

#[cfg(any(feature = "whisper", feature = "zipformer"))]
pub(crate) fn model_path(name: &str) -> PathBuf {
    model_dir().join(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{SinkExt, StreamExt};

    #[test]
    fn version_returns_expected() {
        assert_eq!(version(), "0.1.0");
    }

    #[test]
    fn engine_uses_default_settings() {
        let engine = Processor::new();

        #[cfg(feature = "whisper")]
        assert!(matches!(&engine.backend_kind, AsrBackendKind::Whisper));
        assert!(engine.backend.is_none());
        assert!(!engine.is_ready());
    }

    #[test]
    fn engine_supports_chainable_settings() {
        let engine = Processor::new().set_backend(AsrBackendKind::Whisper);

        #[cfg(feature = "whisper")]
        assert!(matches!(&engine.backend_kind, AsrBackendKind::Whisper));
        assert!(engine.backend.is_none());
        assert!(!engine.is_ready());
    }

    #[tokio::test]
    async fn engine_requires_init_before_sink_use() {
        let mut engine = Processor::new();

        let dummy = AudioFrame {
            samples: vec![],
            channels: 1,
            sample_rate: 16000,
        };
        let err = engine
            .send(dummy)
            .await
            .expect_err("uninitialized engine should fail");

        assert!(matches!(err, AsrError::InvalidInput(_)));
        assert!(!engine.is_ready());
    }

    #[tokio::test]
    async fn engine_requires_init_before_stream_use() {
        let mut engine = Processor::new();

        let err = engine
            .next()
            .await
            .expect("uninitialized engine should yield an error")
            .expect_err("uninitialized engine should fail");

        assert!(matches!(err, AsrError::InvalidInput(_)));
        assert!(!engine.is_ready());
    }
}
