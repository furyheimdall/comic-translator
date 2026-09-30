//! comic-translator batch driver for Koharu's in-process pipeline.
//!
//! Built inside a Koharu 0.83.5 checkout as `koharu-pipeline/src/bin/ct_batch.rs`
//! (see `scripts/setup_koharu.sh`). Processes every image in `--input-dir`
//! (sorted by file name), translates through an OpenAI-compatible endpoint, and
//! writes rendered PNGs with the same stem to `--output-dir`.
//!
//! With `--watch`, the loaded pipeline stays resident and keeps picking up new
//! images that appear in `--input-dir` until the `.closed` marker exists or no
//! new image arrives for `--idle-timeout` seconds.
//!
//! Progress is printed to stdout as `@@CT@@{json}` lines.

use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, bail};
use clap::{Parser, ValueEnum};
use koharu_config::Config;
use koharu_pipeline::{
    Committer, DetectionModel, Flux2KleinConfig, InpaintingModel, KoharuLayoutRFDetrSeg2XLConfig,
    OcrModel, Operation, Pipeline, PipelineConfig, Progress, Request, RoremMixedConfig, Scope,
    StageOutput, TranslationConfig,
};
use koharu_rasterizer::{RasterOptions, Rasterizer};
use koharu_renderer::{Renderer, TypesettingConfig};
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, PageDraft, Session};
use koharu_translator::{GenerationConfig, Language, ModelSelection, Provider, ProvidersConfig};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(about = "Batch-translate a folder of manga pages with Koharu")]
struct Arguments {
    #[arg(long)]
    input_dir: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    /// OpenAI-compatible base URL, e.g. http://127.0.0.1:8710/llm/v1
    #[arg(long)]
    base_url: url::Url,
    /// Model id sent to the endpoint.
    #[arg(long)]
    model: String,
    /// Bearer token; read from CT_LLM_API_KEY when omitted.
    #[arg(long, env = "CT_LLM_API_KEY", hide_env_values = true)]
    api_key: String,
    #[arg(long, value_enum, default_value = "paddleocr-vl-1.6")]
    ocr: OcrChoice,
    #[arg(long, value_enum, default_value = "lama")]
    inpainting: InpaintingChoice,
    #[arg(long, default_value = "ko-KR")]
    target_language: Language,
    #[arg(long)]
    instructions: Option<String>,
    /// Send the page image to the LLM as visual context.
    #[arg(long)]
    vision: bool,
    #[arg(long, default_value_t = 16384)]
    max_tokens: u32,
    /// Font families in priority order.
    #[arg(long = "font-family", default_values_t = vec!["Noto Sans CJK KR".to_owned()])]
    font_families: Vec<String>,
    /// Keep the pipeline loaded and process images added to the input folder later.
    #[arg(long)]
    watch: bool,
    /// In watch mode, exit after this many seconds without a new image.
    #[arg(long, default_value_t = 120)]
    idle_timeout: u64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OcrChoice {
    #[value(name = "paddleocr-vl-1.6")]
    PaddleOcrVl1_6,
    #[value(name = "manga-ocr")]
    MangaOcr,
    #[value(name = "baberu-ocr")]
    BaberuOcr,
    #[value(name = "hayai-ocr")]
    HayaiOcr,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum InpaintingChoice {
    #[value(name = "lama")]
    LaMa,
    #[value(name = "aot-inpainting")]
    AotInpainting,
    #[value(name = "flux2-klein")]
    Flux2Klein,
    #[value(name = "rorem-mixed")]
    RoremMixed,
}

struct SessionCommitter<'a>(&'a mut Session);

#[async_trait::async_trait]
impl Committer for SessionCommitter<'_> {
    async fn commit(&mut self, output: StageOutput) -> Result<koharu_scene::Snapshot> {
        Ok(self.0.commit(output.patch).await?.snapshot)
    }
}

fn emit(progress: f64, message: impl Into<String>) {
    println!("@@CT@@{}", json!({ "progress": progress, "message": message.into() }));
}

fn pipeline_config(arguments: &Arguments) -> PipelineConfig {
    PipelineConfig {
        detection: DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig::default()),
        ocr: match arguments.ocr {
            OcrChoice::PaddleOcrVl1_6 => OcrModel::PaddleOcrVl1_6,
            OcrChoice::MangaOcr => OcrModel::MangaOcr,
            OcrChoice::BaberuOcr => OcrModel::BaberuOcr,
            OcrChoice::HayaiOcr => OcrModel::HayaiOcr,
        },
        translation: TranslationConfig {
            model: ModelSelection {
                provider: Provider::OpenAiCompatible,
                model: Some(arguments.model.clone()),
                quantization: None,
                vision: arguments.vision,
                reasoning: false,
            },
            generation: GenerationConfig {
                max_tokens: Some(arguments.max_tokens),
                vision: Some(arguments.vision),
                reasoning: Some(false),
                ..GenerationConfig::default()
            },
            target_language: arguments.target_language,
            instructions: arguments.instructions.clone(),
        },
        inpainting: match arguments.inpainting {
            InpaintingChoice::LaMa => InpaintingModel::LaMa {},
            InpaintingChoice::AotInpainting => InpaintingModel::AotInpainting {},
            InpaintingChoice::Flux2Klein => InpaintingModel::Flux2Klein(Flux2KleinConfig::default()),
            InpaintingChoice::RoremMixed => InpaintingModel::RoremMixed(RoremMixedConfig::default()),
        },
        processor: Default::default(),
    }
}

fn media_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    }
}

fn list_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut images: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("failed to read {}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.is_file()
                && matches!(
                    path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(),
                    Some("png" | "jpg" | "jpeg" | "webp")
                )
        })
        .collect();
    images.sort();
    Ok(images)
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let images = list_images(&arguments.input_dir)?;
    if images.is_empty() && !arguments.watch {
        bail!("no images in {}", arguments.input_dir.display());
    }
    fs::create_dir_all(&arguments.output_dir)?;

    // Koharu's OpenAI-compatible backend reads its bearer token from the secret store.
    koharu_secrets::set("openai-compatible", &arguments.api_key.as_str().into())
        .context("failed to store the endpoint key in the Linux keyring")?;

    emit(0.0, "Koharu 런타임 초기화 중 (첫 실행 시 CUDA/Torch 패키지와 모델을 내려받습니다)");
    koharu_ml::init().await.context("Koharu runtime initialization failed")?;
    let device = koharu_ml::device(false);

    // `OpenAiCompatibleConfig` is not re-exported; go through the serde shape
    // used by `~/.koharu/config.toml` (`[providers.openai-compatible]`).
    let providers: ProvidersConfig = serde_json::from_value(json!({
        "openai-compatible": { "base_url": arguments.base_url.as_str() }
    }))
    .context("invalid OpenAI-compatible provider configuration")?;
    let pipeline = Pipeline::from_config(
        Config::memory(pipeline_config(&arguments)),
        Config::memory(providers),
        device,
    )?;
    let renderer = Renderer::from_config(Config::memory(TypesettingConfig {
        font_families: arguments.font_families.clone(),
    }))?;
    let rasterizer = Rasterizer::new()?;

    let mut seen = HashSet::new();
    let mut processed = 0usize;
    let mut failures = 0usize;
    let mut idle_since = Instant::now();
    loop {
        let pending: Vec<PathBuf> = list_images(&arguments.input_dir)?
            .into_iter()
            .filter(|path| !seen.contains(path))
            .collect();
        let Some(path) = pending.first().cloned() else {
            if !arguments.watch
                || arguments.input_dir.join(".closed").exists()
                || idle_since.elapsed() >= Duration::from_secs(arguments.idle_timeout)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
            continue;
        };
        seen.insert(path.clone());
        let total = processed + pending.len();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("page").to_owned();
        println!(
            "@@CT@@{}",
            json!({
                "progress": processed as f64 / total as f64,
                "message": format!("페이지 {}/{} 처리 중: {name}", processed + 1, total),
                "page": name,
            })
        );
        let started = Instant::now();
        match process_page(&arguments, &pipeline, &renderer, &rasterizer, &path, processed, total).await {
            Ok(()) => {
                eprintln!("{name} done in {:.1}s", started.elapsed().as_secs_f64());
                emit((processed + 1) as f64 / total as f64, format!("{name} 완료"));
            }
            Err(error) => {
                failures += 1;
                eprintln!("{name} failed: {error:#}");
                println!("@@CT@@{}", json!({ "page_error": name, "message": format!("{error:#}") }));
            }
        }
        processed += 1;
        idle_since = Instant::now();
    }
    emit(1.0, "완료");
    if failures > 0 && !arguments.watch {
        bail!("{failures}/{processed} pages failed");
    }
    Ok(())
}

async fn process_page(
    arguments: &Arguments,
    pipeline: &Pipeline,
    renderer: &Renderer,
    rasterizer: &Rasterizer,
    path: &Path,
    index: usize,
    total: usize,
) -> Result<()> {
    let source = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let decoded = image::load_from_memory(&source).context("failed to decode input image")?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("page");
    let mut session = Session::memory().await?;
    let mut page = None;
    let patch = session.snapshot().patch(|edit| {
        let id = edit.add_page(
            PageDraft::new(name, f64::from(decoded.width()), f64::from(decoded.height())),
            At::End,
        )?;
        edit.set_asset(
            id,
            &AssetRole::new("source")?,
            AssetInput::new(
                Arc::<[u8]>::from(source),
                media_type(path),
                AssetMetadata {
                    width: Some(decoded.width()),
                    height: Some(decoded.height()),
                    attributes: BTreeMap::new(),
                },
            ),
        )?;
        page = Some(id);
        Ok(())
    })?;
    session.commit(patch).await?;
    let page = page.expect("page ID is assigned by the edit");

    let snapshot = session.snapshot();
    let mut committer = SessionCommitter(&mut session);
    let base = index as f64 / total as f64;
    pipeline
        .execute(
            snapshot,
            Request {
                operation: Operation::Full,
                scope: Scope::Pages(vec![page]),
                progress: Some(Arc::new(move |event| {
                    if let Progress::Finished { stage, elapsed, .. } = event {
                        println!(
                            "@@CT@@{}",
                            json!({ "progress": base, "message": format!("{stage} 완료 ({:.1}s)", elapsed.as_secs_f64()) })
                        );
                    }
                })),
                ..Request::default()
            },
            &mut committer,
        )
        .await?;

    let snapshot = session.snapshot();
    let frame = renderer.render(&snapshot, page).await?;
    let raster = rasterizer.rasterize(&frame.raster_frame()?, RasterOptions::default())?;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("page");
    let output = arguments.output_dir.join(format!("{stem}.png"));
    raster
        .image
        .save(&output)
        .with_context(|| format!("failed to write {}", output.display()))?;
    Ok(())
}
