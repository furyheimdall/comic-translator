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
//! `--cache-dir` stores immutable `<stem>/{manifest.json,session/}` checkpoints
//! after detection, OCR and inpainting, before any translation. Copy the whole
//! entry (including every session blob) to reuse it under another page stem.
//! Entries and temporary working sessions must remain private to the owning job.
//!
//! Progress is printed to stdout as `@@CT@@{json}` lines.

use std::{
    collections::{BTreeMap, HashSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use koharu_config::Config;
use koharu_pipeline::{
    Committer, DetectionModel, Flux2KleinConfig, InpaintingModel, KoharuLayoutRFDetrSeg2XLConfig,
    OcrModel, Operation, Pipeline, PipelineConfig, Progress, Request, RoremMixedConfig, Scope,
    RunStatus, Stage, StageOutput, TranslationConfig,
};
use koharu_rasterizer::{RasterOptions, Rasterizer};
use koharu_renderer::{Renderer, TypesettingConfig};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, EntityId, PageDraft, RasterLayer, RasterLayerKind,
    Session, Snapshot,
};
use koharu_translator::{GenerationConfig, Language, ModelSelection, Provider, ProvidersConfig};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Batch-translate a folder of manga pages with Koharu")]
struct Arguments {
    #[arg(long)]
    input_dir: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    /// Private intermediate checkpoints; omit to retain the ordinary full pipeline.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
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

// Bump when the pinned engine or our preprocessing patch changes semantics.
const PREPROCESSING_REVISION: &str = "koharu-0.83.5-ct-1";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct CheckpointKey {
    schema_version: u32,
    pipeline_revision: String,
    input_sha256: String,
    ocr: String,
    inpainting: String,
    preprocessing_sha256: String,
}

impl CheckpointKey {
    fn new(arguments: &Arguments, source: &[u8]) -> Result<Self> {
        let mut config = serde_json::to_value(pipeline_config(arguments))?;
        config.as_object_mut().context("pipeline config is not an object")?.remove("translation");
        Ok(Self {
            schema_version: 1,
            pipeline_revision: PREPROCESSING_REVISION.to_owned(),
            input_sha256: format!("{:x}", Sha256::digest(source)),
            ocr: arguments.ocr.to_possible_value().expect("OCR choice").get_name().to_owned(),
            inpainting: arguments.inpainting.to_possible_value().expect("inpainting choice").get_name().to_owned(),
            preprocessing_sha256: format!("{:x}", Sha256::digest(serde_json::to_vec(&config)?)),
        })
    }
}

#[derive(Deserialize, Serialize)]
struct CheckpointManifest {
    #[serde(flatten)]
    key: CheckpointKey,
    // Paths are relative to the entry, including the session/ prefix. Never use
    // manifest paths for filesystem access: enumerate the actual tree instead.
    files: BTreeMap<String, String>,
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn copy_session_tree(
    source: &Path,
    destination: &Path,
    relative: &Path,
    files: &mut BTreeMap<String, String>,
    durable: bool,
) -> Result<()> {
    ensure!(fs::symlink_metadata(source)?.is_dir(), "checkpoint session is not a directory");
    fs::create_dir(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let target = destination.join(entry.file_name());
        let relative = relative.join(entry.file_name());
        if kind.is_dir() {
            copy_session_tree(&entry.path(), &target, &relative, files, durable)?;
        } else {
            ensure!(kind.is_file(), "checkpoint contains a non-regular file");
            fs::copy(entry.path(), &target)?;
            if durable {
                fs::File::open(&target)?.sync_all()?;
            }
            files.insert(
                relative.to_str().context("non-UTF-8 checkpoint path")?.to_owned(),
                file_sha256(&target)?,
            );
        }
    }
    if durable {
        fs::File::open(destination)?.sync_all()?;
    }
    Ok(())
}

async fn checkpoint_page(snapshot: &Snapshot, key: &CheckpointKey) -> Result<EntityId> {
    let mut pages = snapshot.pages();
    let page = pages.next().context("checkpoint has no page")?.id();
    ensure!(pages.next().is_none(), "checkpoint has more than one page");
    let source_role = AssetRole::new("source")?;
    let source = snapshot.asset(page, &source_role)?.context("checkpoint has no source image")?;
    ensure!(
        format!("{:x}", Sha256::digest(snapshot.read_blob(source.blob).await?)) == key.input_sha256,
        "checkpoint source image does not match the input"
    );
    let mut cleanup = false;
    for entity in snapshot.children(page)? {
        if snapshot.component::<RasterLayer>(entity)?
            .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
            && snapshot.asset(entity, &source_role)?.is_some()
        {
            cleanup = true;
        }
    }
    ensure!(cleanup, "checkpoint has no erased image layer");
    Ok(page)
}

async fn restore_checkpoint(
    entry: &Path,
    working: &Path,
    key: &CheckpointKey,
) -> Result<(Session, EntityId)> {
    let manifest: CheckpointManifest = serde_json::from_slice(&fs::read(entry.join("manifest.json"))?)?;
    ensure!(manifest.key == *key, "checkpoint input or preprocessing settings changed");
    ensure!(!manifest.files.is_empty(), "checkpoint has no session files");
    let mut files = BTreeMap::new();
    copy_session_tree(
        &entry.join("session"), working, Path::new("session"), &mut files, false,
    )?;
    ensure!(files == manifest.files, "checkpoint session files are missing or corrupted");
    let session = Session::open(working).await?;
    let page = checkpoint_page(&session.snapshot(), key).await?;
    Ok((session, page))
}

fn publish_checkpoint(
    entry: &Path,
    working: &Path,
    key: &CheckpointKey,
) -> Result<()> {
    let parent = entry.parent().context("checkpoint has no parent")?;
    let staging = tempfile::Builder::new().prefix(".checkpoint-").tempdir_in(parent)?;
    let mut files = BTreeMap::new();
    copy_session_tree(
        working, &staging.path().join("session"), Path::new("session"), &mut files, true,
    )?;
    let manifest = CheckpointManifest { key: key.clone(), files };
    let mut file = fs::File::create(staging.path().join("manifest.json"))?;
    file.write_all(&serde_json::to_vec(&manifest)?)?;
    file.sync_all()?;
    fs::File::open(staging.path())?.sync_all()?;
    // Readers only accept a complete directory. A crash while replacing an old
    // incompatible entry may cause a miss, never a partially published hit.
    if entry.exists() {
        fs::remove_dir_all(entry)?;
    }
    fs::rename(staging.path(), entry)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

async fn execute_stages(
    pipeline: &Pipeline,
    session: &mut Session,
    page: EntityId,
    operation: Operation,
    base: f64,
) -> Result<()> {
    let snapshot = session.snapshot();
    let mut committer = SessionCommitter(session);
    let report = pipeline.execute(
        snapshot,
        Request {
            operation,
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
    ).await?;
    ensure!(report.status == RunStatus::Completed, "pipeline stopped before completion");
    Ok(())
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
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("page");
    let base = index as f64 / total as f64;
    let mut cache = arguments.cache_dir.as_ref().and_then(|root| {
        let prepare = || -> Result<_> {
            ensure!(!matches!(stem, "." | ".."), "unsafe checkpoint stem");
            fs::create_dir_all(root)?;
            let working = tempfile::Builder::new().prefix(".work-").tempdir_in(root)?;
            Ok((working, root.join(stem), CheckpointKey::new(arguments, &source)?))
        };
        match prepare() {
            Ok(cache) => Some(cache),
            // Cache diagnostics would be copied into the user-visible job log.
            Err(_) => None,
        }
    });
    let restored = if let Some((working, entry, key)) = &cache {
        let working = working.path().join("session");
        match restore_checkpoint(entry, &working, key).await {
            Ok(restored) => Some(restored),
            Err(_) => {
                // Absent, incompatible or damaged entries are ordinary misses.
                if working.exists() && fs::remove_dir_all(&working).is_err() {
                    cache = None;
                }
                None
            }
        }
    } else {
        None
    };
    let mut translate_only = restored.is_some();
    let (mut session, page) = if let Some(restored) = restored {
        restored
    } else {
        let mut session = if let Some((working, _, _)) = &cache {
            match Session::create(working.path().join("session")).await {
                Ok(session) => session,
                Err(_) => {
                    cache = None;
                    Session::memory().await?
                }
            }
        } else {
            Session::memory().await?
        };
        let decoded = image::load_from_memory(&source).context("failed to decode input image")?;
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("page");
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
        if let Some((working, entry, key)) = &cache {
            execute_stages(
                pipeline, &mut session, page,
                Operation::Stages { stages: vec![Stage::Detection, Stage::Ocr, Stage::Inpainting] },
                base,
            ).await?;
            translate_only = true;
            // Never translate the published session: later requests must start
            // from untranslated OCR, even if this request's translation fails.
            let _ = async {
                checkpoint_page(&session.snapshot(), key).await?;
                publish_checkpoint(entry, &working.path().join("session"), key)
            }.await;
        }
        (session, page)
    };
    execute_stages(
        pipeline, &mut session, page,
        if translate_only { Operation::Only { stage: Stage::Translation } } else { Operation::Full },
        base,
    ).await?;

    let snapshot = session.snapshot();
    let frame = renderer.render(&snapshot, page).await?;
    let raster = rasterizer.rasterize(&frame.raster_frame()?, RasterOptions::default())?;
    let output = arguments.output_dir.join(format!("{stem}.png"));
    raster
        .image
        .save(&output)
        .with_context(|| format!("failed to write {}", output.display()))?;
    Ok(())
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use koharu_scene::{Authored, Origin, SourceText, Translation};

    fn arguments() -> Arguments {
        Arguments::try_parse_from([
            "ct_batch", "--input-dir", ".", "--output-dir", ".",
            "--base-url", "http://localhost:8710/v1", "--model", "first",
            "--api-key", "test",
        ]).unwrap()
    }

    #[test]
    fn checkpoint_identity_ignores_translation_but_not_image_processing() {
        let mut args = arguments();
        let original = CheckpointKey::new(&args, b"source").unwrap();
        args.model = "different".to_owned();
        args.instructions = Some("different style".to_owned());
        args.vision = true;
        args.max_tokens = 2048;
        args.target_language = Language::English;
        args.font_families = vec!["different font".to_owned()];
        assert_eq!(original, CheckpointKey::new(&args, b"source").unwrap());
        assert_ne!(original, CheckpointKey::new(&args, b"changed").unwrap());
        args.ocr = OcrChoice::MangaOcr;
        assert_ne!(original, CheckpointKey::new(&args, b"source").unwrap());
        args.ocr = OcrChoice::PaddleOcrVl1_6;
        args.inpainting = InpaintingChoice::AotInpainting;
        assert_ne!(original, CheckpointKey::new(&args, b"source").unwrap());
    }

    #[tokio::test]
    async fn checkpoint_preserves_pixels_and_ocr_without_persisting_translation() -> Result<()> {
        let root = tempfile::tempdir()?;
        let working = root.path().join("working");
        let entry = root.path().join("page-one");
        let mut encoded = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(2, 2).write_to(&mut encoded, image::ImageFormat::Png)?;
        let source = encoded.into_inner();
        let key = CheckpointKey::new(&arguments(), &source)?;
        let mut session = Session::create(&working).await?;
        let mut text = None;
        let mut cleanup = None;
        let patch = session.snapshot().patch(|edit| {
            let page = edit.add_page(PageDraft::new("original.png", 2.0, 2.0), At::End)?;
            let layer = edit.add_entity(page, At::Start)?;
            edit.set(layer, &RasterLayer {
                origin: Origin::User,
                name: "Cleanup".to_owned(),
                kind: RasterLayerKind::Cleanup,
            })?;
            for entity in [page, layer] {
                edit.set_asset(entity, &AssetRole::new("source")?, AssetInput::new(
                    Arc::<[u8]>::from(source.clone()),
                    "image/png",
                    AssetMetadata { width: Some(2), height: Some(2), attributes: BTreeMap::new() },
                ))?;
            }
            let content = edit.add_text_content(page, At::End)?;
            edit.set(content, &SourceText { text: Authored::user("原文".to_owned()), language: None })?;
            text = Some(content);
            cleanup = Some(layer);
            Ok(())
        })?;
        session.commit(patch).await?;
        checkpoint_page(&session.snapshot(), &key).await?;
        publish_checkpoint(&entry, &working, &key)?;

        // A different stem is supported, and translating the working copy must
        // never leak a previous model's text back into the reusable checkpoint.
        let renamed = root.path().join("page-two");
        fs::rename(&entry, &renamed)?;
        let (mut restored, _) = restore_checkpoint(&renamed, &root.path().join("first"), &key).await?;
        let text = text.unwrap();
        let patch = restored.snapshot().patch(|edit| {
            edit.set(text, &Translation { text: Authored::user("translated".to_owned()), language: None })
        })?;
        restored.commit(patch).await?;
        let (fresh, _) = restore_checkpoint(&renamed, &root.path().join("second"), &key).await?;
        let snapshot = fresh.snapshot();
        assert!(snapshot.component::<Translation>(text)?.is_none());
        assert_eq!(snapshot.component::<SourceText>(text)?.unwrap().text.value, "原文");
        let erased = snapshot.asset(cleanup.unwrap(), &AssetRole::new("source")?)?.unwrap();
        assert_eq!(snapshot.read_blob(erased.blob).await?.as_ref(), source.as_slice());

        let changed_key = CheckpointKey::new(&arguments(), b"different image")?;
        assert!(restore_checkpoint(&renamed, &root.path().join("mismatch"), &changed_key).await.is_err());
        let manifest: CheckpointManifest = serde_json::from_slice(&fs::read(renamed.join("manifest.json"))?)?;
        let blob = manifest.files.keys().find(|path| path.starts_with("session/blobs/")).unwrap();
        fs::write(renamed.join(blob), b"corrupt")?;
        assert!(restore_checkpoint(&renamed, &root.path().join("corrupt"), &key).await.is_err());
        fs::remove_file(renamed.join(blob))?;
        assert!(restore_checkpoint(&renamed, &root.path().join("partial"), &key).await.is_err());
        Ok(())
    }
}
