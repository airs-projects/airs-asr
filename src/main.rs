use std::error::Error;
use std::fs::File;
use std::io;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::pin::Pin;
use std::process::ExitCode;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::task::{Context, Poll};
use std::thread;

use airs_asr::{AsrBackendKind, Processor};
use airs_audio::AudioStream;
use futures::{Sink, SinkExt, StreamExt};

type AppResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone)]
struct PipeOptions {
    source: SourceSpec,
    targets: Vec<TargetSpec>,
    backend: AsrBackendKind,
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum SourceSpec {
    Device(Option<String>),
    File(PathBuf),
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum TargetSpec {
    File(PathBuf),
    Stdout,
}

struct TextSink {
    target: TargetSpec,
    file: Option<BufWriter<File>>,
}

#[derive(Debug, Clone)]
struct AsrDefaults {
    backend: AsrBackendKind,
}

impl Default for AsrDefaults {
    fn default() -> Self {
        Self {
            backend: AsrBackendKind::Whisper,
        }
    }
}

#[derive(Debug)]
enum Command {
    Help,
    Version,
    Pipe { options: PipeOptions },
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> AppResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match parse_command(args)? {
        Command::Help => cmd_help(),
        Command::Version => cmd_version(),
        Command::Pipe { options } => cmd_pipe(options).await?,
    }

    Ok(())
}

fn parse_command(args: Vec<String>) -> Result<Command, io::Error> {
    match args.as_slice() {
        [] => Ok(Command::Help),
        [arg] if arg == "--help" => Ok(Command::Help),
        [arg] if arg == "--version" => Ok(Command::Version),
        [command, ..] if command == "pipe" => parse_pipe(&args[1..]),
        [command, ..] => Err(invalid(format!("unknown command: {command}"))),
    }
}

fn parse_pipe(args: &[String]) -> Result<Command, io::Error> {
    let defaults = AsrDefaults::default();
    let mut source = None;
    let mut targets = Vec::new();
    let mut backend = defaults.backend;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-i" => {
                if source.is_some() {
                    return Err(invalid("-i can only be used once"));
                }
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| invalid("-i requires file:<path> or device[:name]"))?;
                source = Some(parse_source(value)?);
            }
            "-o" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| invalid("-o requires file:<path> or stdout"))?;
                targets.push(parse_target(value)?);
            }
            "--backend" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| invalid("--backend requires a name"))?;
                backend = parse_backend(value)?;
            }
            arg => return Err(invalid(format!("unexpected argument: {arg}"))),
        }
        i += 1;
    }

    let source =
        source.ok_or_else(|| invalid("pipe requires -i file:<path> or -i device[:name]"))?;
    if targets.is_empty() {
        return Err(invalid(
            "pipe requires at least one -o file:<path> or -o stdout",
        ));
    }

    Ok(Command::Pipe {
        options: PipeOptions {
            source,
            targets,
            backend,
        },
    })
}

fn parse_source(value: &str) -> Result<SourceSpec, io::Error> {
    match split_typed_value(value) {
        ("file", Some(path)) if !path.is_empty() => Ok(SourceSpec::File(PathBuf::from(path))),
        ("file", _) => Err(invalid("-i file requires a path")),
        ("device", name) => Ok(SourceSpec::Device(
            name.filter(|name| !name.is_empty()).map(str::to_owned),
        )),
        (kind, _) => Err(invalid(format!("unsupported input type: {kind}"))),
    }
}

fn parse_target(value: &str) -> Result<TargetSpec, io::Error> {
    match split_typed_value(value) {
        ("file", Some(path)) if !path.is_empty() => Ok(TargetSpec::File(PathBuf::from(path))),
        ("file", _) => Err(invalid("-o file requires a path")),
        ("stdout", None) => Ok(TargetSpec::Stdout),
        ("stdout", Some("")) => Ok(TargetSpec::Stdout),
        ("stdout", Some(_)) => Err(invalid("-o stdout does not accept a value")),
        (kind, _) => Err(invalid(format!("unsupported output type: {kind}"))),
    }
}

fn split_typed_value(value: &str) -> (&str, Option<&str>) {
    match value.split_once(':') {
        Some((kind, value)) => (kind, Some(value)),
        None => (value, None),
    }
}

fn cmd_help() {
    let defaults = AsrDefaults::default();
    println!("Usage:");
    println!("  airs-asr --help");
    println!("  airs-asr --version");
    println!("  airs-asr pipe -i <source> -o <target> [-o <target>...] [--backend <name>]");
    println!();
    println!("Source:");
    println!("  -i file:<path>    Audio file");
    println!("  -i device         Default microphone device");
    println!("  -i device:<name>  Named microphone device");
    println!();
    println!("Target:");
    println!("  -o file:<path>    Text file");
    println!("  -o stdout         Standard output");
    println!();
    println!("Defaults: backend={}.", backend_name(defaults.backend));
}

fn cmd_version() {
    println!("{}", airs_asr::version());
}

async fn cmd_pipe(options: PipeOptions) -> AppResult<()> {
    let mut asr = Processor::new().set_backend(options.backend).init().await?;

    let is_device_source = matches!(options.source, SourceSpec::Device(_));
    if is_device_source {
        wait_for_enter("Press Enter to start recording.")?;
    }

    let mut input = options.source.clone().into_stream();
    if is_device_source {
        input.start()?;
    }
    let stop_recording = if is_device_source {
        eprintln!("Recording. Press Enter to stop.");
        Some(spawn_enter_signal())
    } else {
        None
    };

    let mut outputs = options
        .targets
        .iter()
        .cloned()
        .map(TextSink::new)
        .collect::<Vec<_>>();

    while !stop_recording
        .as_ref()
        .is_some_and(|stop| stop.load(Ordering::SeqCst))
    {
        let Some(frame) = input.next().await else {
            break;
        };
        let frame = frame?;
        asr.send(frame).await?;
    }
    asr.close().await?;

    while let Some(result) = asr.next().await {
        let text = result?;
        for output in outputs.iter_mut() {
            output.send(text.clone()).await?;
        }
    }

    for output in &mut outputs {
        output.close().await?;
    }

    Ok(())
}

impl SourceSpec {
    fn into_stream(self) -> AudioStream {
        match self {
            Self::Device(name) => AudioStream::from_device(name),
            Self::File(path) => AudioStream::from_file(path),
        }
    }
}

impl TextSink {
    fn new(target: TargetSpec) -> Self {
        Self { target, file: None }
    }
}

impl Sink<String> for TextSink {
    type Error = io::Error;

    fn poll_ready(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(mut self: Pin<&mut Self>, item: String) -> io::Result<()> {
        match &self.target {
            TargetSpec::Stdout => {
                println!("{item}");
                Ok(())
            }
            TargetSpec::File(path) => {
                if self.file.is_none() {
                    self.file = Some(BufWriter::new(File::create(path)?));
                }
                let file = self.file.as_mut().expect("text output file is initialized");
                writeln!(file, "{item}")
            }
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.file.as_mut() {
            Some(file) => Poll::Ready(file.flush()),
            None => Poll::Ready(Ok(())),
        }
    }

    fn poll_close(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
        if let Some(mut file) = self.file.take() {
            file.flush()?;
        }
        Poll::Ready(Ok(()))
    }
}

fn wait_for_enter(message: &str) -> io::Result<()> {
    eprintln!("{message}");
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(())
}

fn spawn_enter_signal() -> Arc<AtomicBool> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_signal = stop.clone();
    thread::spawn(move || {
        let mut line = String::new();
        let _ = io::stdin().read_line(&mut line);
        stop_signal.store(true, Ordering::SeqCst);
    });
    stop
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn parse_backend(name: &str) -> Result<AsrBackendKind, io::Error> {
    match name {
        "whisper" => Ok(AsrBackendKind::Whisper),
        "zipformer" => Ok(AsrBackendKind::Zipformer),
        _ => Err(invalid(format!("unsupported backend: {name}"))),
    }
}

fn backend_name(backend: AsrBackendKind) -> &'static str {
    match backend {
        AsrBackendKind::Whisper => "whisper",
        AsrBackendKind::Zipformer => "zipformer",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pipe_file_to_stdout() {
        let cmd = parse_command(vec![
            "pipe".to_string(),
            "-i".to_string(),
            "file:audio.wav".to_string(),
            "-o".to_string(),
            "stdout".to_string(),
        ])
        .expect("parse command");

        match cmd {
            Command::Pipe { options } => {
                assert!(matches!(options.source, SourceSpec::File(_)));
                assert!(matches!(options.targets.as_slice(), [TargetSpec::Stdout]));
            }
            _ => panic!("expected Pipe"),
        }
    }

    #[test]
    fn parse_pipe_device_to_file() {
        let cmd = parse_command(vec![
            "pipe".to_string(),
            "-i".to_string(),
            "device".to_string(),
            "-o".to_string(),
            "file:out.txt".to_string(),
        ])
        .expect("parse command");

        match cmd {
            Command::Pipe { options } => {
                assert!(matches!(options.source, SourceSpec::Device(None)));
                assert_eq!(options.targets.len(), 1);
            }
            _ => panic!("expected Pipe"),
        }
    }

    #[test]
    fn parse_pipe_named_device_to_file() {
        let cmd = parse_command(vec![
            "pipe".to_string(),
            "-i".to_string(),
            "device:Microphone".to_string(),
            "-o".to_string(),
            "file:out.txt".to_string(),
        ])
        .expect("parse command");

        match cmd {
            Command::Pipe { options } => {
                assert!(matches!(
                    options.source,
                    SourceSpec::Device(Some(ref name)) if name == "Microphone"
                ));
            }
            _ => panic!("expected Pipe"),
        }
    }

    #[test]
    fn parse_pipe_multiple_targets() {
        let cmd = parse_command(vec![
            "pipe".to_string(),
            "-i".to_string(),
            "file:audio.wav".to_string(),
            "-o".to_string(),
            "stdout".to_string(),
            "-o".to_string(),
            "file:out.txt".to_string(),
        ])
        .expect("parse command");

        match cmd {
            Command::Pipe { options } => {
                assert_eq!(options.targets.len(), 2);
            }
            _ => panic!("expected Pipe"),
        }
    }

    #[test]
    fn parse_pipe_missing_source_fails() {
        let err = parse_command(vec![
            "pipe".to_string(),
            "-o".to_string(),
            "stdout".to_string(),
        ])
        .expect_err("missing source should fail");

        assert_eq!(
            err.to_string(),
            "pipe requires -i file:<path> or -i device[:name]"
        );
    }

    #[test]
    fn parse_pipe_missing_target_fails() {
        let err = parse_command(vec![
            "pipe".to_string(),
            "-i".to_string(),
            "file:audio.wav".to_string(),
        ])
        .expect_err("missing target should fail");

        assert_eq!(
            err.to_string(),
            "pipe requires at least one -o file:<path> or -o stdout"
        );
    }

    #[test]
    fn parse_pipe_backend() {
        let cmd = parse_command(vec![
            "pipe".to_string(),
            "-i".to_string(),
            "file:audio.wav".to_string(),
            "-o".to_string(),
            "stdout".to_string(),
            "--backend".to_string(),
            "whisper".to_string(),
        ])
        .expect("parse command");

        match cmd {
            Command::Pipe { options } => {
                assert_eq!(options.backend, AsrBackendKind::Whisper);
            }
            _ => panic!("expected Pipe"),
        }
    }

    #[test]
    fn parse_pipe_zipformer_backend() {
        let cmd = parse_command(vec![
            "pipe".to_string(),
            "-i".to_string(),
            "file:audio.wav".to_string(),
            "-o".to_string(),
            "stdout".to_string(),
            "--backend".to_string(),
            "zipformer".to_string(),
        ])
        .expect("parse command");

        match cmd {
            Command::Pipe { options } => {
                assert_eq!(options.backend, AsrBackendKind::Zipformer);
            }
            _ => panic!("expected Pipe"),
        }
    }

    #[test]
    fn parse_pipe_windows_file_paths() {
        let cmd = parse_command(vec![
            "pipe".to_string(),
            "-i".to_string(),
            "file:E:\\audio.wav".to_string(),
            "-o".to_string(),
            "file:E:\\out.txt".to_string(),
        ])
        .expect("parse command");

        match cmd {
            Command::Pipe { options } => {
                assert_eq!(
                    options.source,
                    SourceSpec::File(PathBuf::from("E:\\audio.wav"))
                );
                assert_eq!(
                    options.targets,
                    vec![TargetSpec::File(PathBuf::from("E:\\out.txt"))]
                );
            }
            _ => panic!("expected Pipe"),
        }
    }
}
