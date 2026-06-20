use std::collections::VecDeque;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use airs_audio::AudioFrame;
use async_trait::async_trait;
use futures::{Sink, Stream};
use sherpa_onnx::{
    OnlineRecognizer, OnlineRecognizerConfig, OnlineStream, OnlineTransducerModelConfig,
};

use crate::backends::{AsrBackend, BackendSessionStream};
use crate::{AsrError, Result};

/// Streaming Zipformer speech recognition engine backed by sherpa-onnx.
pub(crate) struct ZipformerEngine {
    recognizer: Option<Arc<OnlineRecognizer>>,
    stream: Option<BackendSessionStream>,
}

impl ZipformerEngine {
    pub fn new() -> Self {
        Self {
            recognizer: None,
            stream: None,
        }
    }

    fn find_file(dir: &PathBuf, exact: &str, prefix: &str) -> Option<PathBuf> {
        let path = dir.join(exact);
        if path.exists() {
            return Some(path);
        }

        let check = |d: &PathBuf| -> Option<PathBuf> {
            if let Ok(entries) = std::fs::read_dir(d) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    if name.starts_with(prefix) && name.ends_with(".onnx") {
                        return Some(path);
                    }
                    if name == exact {
                        return Some(path);
                    }
                }
            }
            None
        };

        if let Some(found) = check(dir) {
            return Some(found);
        }

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let child = entry.path();
                if child.is_dir() {
                    let nested = child.join(exact);
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

    fn ensure_stream(&mut self) -> Result<&mut BackendSessionStream> {
        if self.stream.is_none() {
            let recognizer = self.recognizer.as_ref().ok_or_else(|| {
                AsrError::InvalidInput(
                    "Zipformer backend not initialized. Call init() first.".into(),
                )
            })?;

            self.stream = Some(Box::pin(ZipformerStream::new(recognizer.clone())));
        }

        Ok(self.stream.as_mut().expect("ASR stream is initialized"))
    }
}

impl Default for ZipformerEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ZipformerEngine {
    /// One-shot: transcribe a single audio frame synchronously.
    pub fn process(&self, audio: &AudioFrame) -> Result<String> {
        let recognizer = self.recognizer.as_ref().ok_or_else(|| {
            AsrError::InvalidInput("Zipformer backend not initialized. Call init() first.".into())
        })?;

        let stream = recognizer.create_stream();
        let samples = crate::backends::mono_samples(audio)?;
        stream.accept_waveform(audio.sample_rate as i32, &samples);
        stream.input_finished();

        while recognizer.is_ready(&stream) {
            recognizer.decode(&stream);
        }

        let result = recognizer
            .get_result(&stream)
            .ok_or_else(|| AsrError::Transcription("no recognition result".into()))?;

        Ok(result.text)
    }
}

#[async_trait]
impl AsrBackend for ZipformerEngine {
    async fn init(&mut self) -> Result<()> {
        let model_dir = crate::model_path("zipformer");

        let encoder = Self::find_file(&model_dir, "encoder.onnx", "encoder").ok_or_else(|| {
            AsrError::BackendLoad(format!(
                "encoder.onnx (or encoder*.onnx) not found in {}",
                model_dir.display()
            ))
        })?;

        let decoder = Self::find_file(&model_dir, "decoder.onnx", "decoder").ok_or_else(|| {
            AsrError::BackendLoad(format!(
                "decoder.onnx (or decoder*.onnx) not found in {}",
                model_dir.display()
            ))
        })?;

        let joiner = Self::find_file(&model_dir, "joiner.onnx", "joiner").ok_or_else(|| {
            AsrError::BackendLoad(format!(
                "joiner.onnx (or joiner*.onnx) not found in {}",
                model_dir.display()
            ))
        })?;

        let tokens = Self::find_file(&model_dir, "tokens.txt", "tokens").ok_or_else(|| {
            AsrError::BackendLoad(format!("tokens.txt not found in {}", model_dir.display()))
        })?;

        let mut config = OnlineRecognizerConfig::default();
        config.model_config.transducer = OnlineTransducerModelConfig {
            encoder: Some(Self::to_path_string(&encoder)),
            decoder: Some(Self::to_path_string(&decoder)),
            joiner: Some(Self::to_path_string(&joiner)),
        };
        config.model_config.tokens = Some(Self::to_path_string(&tokens));
        config.model_config.num_threads = 4;
        config.model_config.provider = Some("cpu".into());
        config.decoding_method = Some("greedy_search".into());

        let recognizer = OnlineRecognizer::create(&config)
            .ok_or_else(|| AsrError::BackendLoad("failed to create Zipformer recognizer".into()))?;

        self.recognizer = Some(Arc::new(recognizer));
        Ok(())
    }

    async fn process(&mut self, audio: AudioFrame) -> Result<String> {
        ZipformerEngine::process(self, &audio)
    }
}

impl Sink<AudioFrame> for ZipformerEngine {
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

impl Stream for ZipformerEngine {
    type Item = Result<String>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.ensure_stream() {
            Ok(stream) => stream.as_mut().poll_next(context),
            Err(error) => Poll::Ready(Some(Err(error))),
        }
    }
}

struct ZipformerStream {
    recognizer: Arc<OnlineRecognizer>,
    stream: OnlineStream,
    pending: VecDeque<Result<String>>,
    last_text: String,
    closed: bool,
    waker: Option<Waker>,
}

impl ZipformerStream {
    fn new(recognizer: Arc<OnlineRecognizer>) -> Self {
        let stream = recognizer.create_stream();
        Self {
            recognizer,
            stream,
            pending: VecDeque::new(),
            last_text: String::new(),
            closed: false,
            waker: None,
        }
    }

    fn decode_ready(&mut self) {
        while self.recognizer.is_ready(&self.stream) {
            self.recognizer.decode(&self.stream);
        }

        self.emit_current_result();
    }

    fn emit_current_result(&mut self) {
        let Some(result) = self.recognizer.get_result(&self.stream) else {
            return;
        };

        if result.text.is_empty() || result.text == self.last_text {
            return;
        }

        self.last_text = result.text.clone();
        self.pending.push_back(Ok(result.text));
        if let Some(waker) = self.waker.take() {
            waker.wake();
        }
    }
}

impl Sink<AudioFrame> for ZipformerStream {
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

        let samples = crate::backends::mono_samples(&item)?;
        self.stream
            .accept_waveform(item.sample_rate as i32, &samples);
        self.decode_ready();
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
            self.stream.input_finished();
            self.decode_ready();
        }

        Poll::Ready(Ok(()))
    }
}

impl Stream for ZipformerStream {
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
    fn find_file_accepts_prefixed_model_names() {
        let dir =
            std::env::temp_dir().join(format!("airs-asr-zipformer-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let encoder = dir.join("encoder-epoch-99-avg-1.int8.onnx");
        std::fs::write(&encoder, "").unwrap();

        assert_eq!(
            ZipformerEngine::find_file(&dir, "encoder.onnx", "encoder"),
            Some(encoder)
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
