use std::pin::Pin;

use airs_audio::AudioFrame;
use async_trait::async_trait;
use futures::{Sink, Stream};

use crate::{AsrError, Result};

#[cfg(feature = "whisper")]
pub(crate) mod whisper;

#[cfg(feature = "zipformer")]
pub(crate) mod zipformer;

pub(crate) trait BackendSession:
    Sink<AudioFrame, Error = AsrError> + Stream<Item = Result<String>> + Send + Unpin
{
}

impl<T> BackendSession for T where
    T: Sink<AudioFrame, Error = AsrError> + Stream<Item = Result<String>> + Send + Unpin
{
}

pub(crate) type BackendSessionStream = Pin<Box<dyn BackendSession>>;

#[async_trait]
pub trait AsrBackend:
    Sink<AudioFrame, Error = AsrError> + Stream<Item = Result<String>> + Send + Unpin
{
    /// Load the model and prepare for transcription.
    async fn init(&mut self) -> Result<()>;

    /// One-shot: transcribe a single audio frame and return the recognized text.
    async fn process(&mut self, audio: AudioFrame) -> Result<String>;
}

#[cfg(any(feature = "whisper", feature = "zipformer"))]
pub(crate) fn mono_samples(audio: &AudioFrame) -> Result<Vec<f32>> {
    if audio.channels == 0 {
        return Err(AsrError::InvalidInput(
            "audio channel count is missing".into(),
        ));
    }

    let channels = audio.channels as usize;
    if audio.samples.len() % channels != 0 {
        return Err(AsrError::InvalidInput(
            "audio samples are not aligned to channels".into(),
        ));
    }

    if channels == 1 {
        return Ok(audio.samples.clone());
    }

    Ok(audio
        .samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect())
}
