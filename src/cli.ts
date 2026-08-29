import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { copyFile, mkdir, writeFile } from 'node:fs/promises';
import { extname, isAbsolute, join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { ZodError } from 'zod';
import { registerAllAdapters } from './adapters/register.js';
import { parseTimeToSeconds } from './processors/annotated-timeline.js';
import { renderTranscriptText } from './processors/slide-sync.js';
import {
  AnalyzeOptionsSchema,
  assembleResultDoc,
  getAnalysis,
  resolveAnalyzeParams,
} from './tools/analyze-core.js';
import type { AnalyzeOptions, ProgressReporter } from './tools/analyze-core.js';
import type { IAnalysisResult, IFrameResult, ISlideNarration } from './types.js';
import { persistentCacheDir } from './utils/temp-files.js';
import { isVideoSource } from './utils/url-detector.js';

/** Subdirectory of `--out` holding the slide images and slides.json. */
const SLIDES_DIRNAME = 'slides';

const CLI_USAGE = `Usage: mcp-video-analyzer analyze <url-or-path> [options]

One-shot video analysis. Prints a single JSON document to stdout (progress and
errors go to stderr). Frame images are copied to --out and referenced by
absolute path in the JSON. Partial failures degrade into the "warnings" array
(exit 0); only a hard failure exits 1.

Options:
  --detail <level>        brief | standard | detailed (default: standard)
  --max-frames <n>        Max key frames to extract (1-60; default adapts to duration)
  --max-width <px>        Width cap for emitted frames (default: 800, or
                          MCP_FRAME_MAX_WIDTH). 0 keeps the source resolution —
                          use it for screen recordings whose meaning lives in
                          small text (terminals, dashboards, IDEs)
  --fields <list>         Output filter — comma-separated subset of: metadata,
                          transcript,frames,comments,chapters,ocrResults,
                          timeline,aiSummary. Filters the emitted JSON only;
                          use --detail brief to actually skip frame extraction.
  --force-refresh         Bypass cache and re-analyze
  --ocr-language <codes>  Tesseract OCR languages (default: eng+por)
  --model <name>          Whisper model override (e.g. small, medium)
  --language <code>       Forced transcription language (e.g. pt)
  --out <dir>             Where to copy frame images
                          (default: <tmp>/mcp-video-analyzer/<url-hash>)
  -h, --help              Show this help

Sources: Loom, YouTube, Vimeo, TikTok, Instagram, X/Twitter, Twitch,
Dailymotion, Facebook (yt-dlp required), direct .mp4/.webm/.mov URLs, and
local paths / file:// URIs.
`;

export interface CliInvocation {
  url: string | undefined;
  options: AnalyzeOptions;
  outDir: string | undefined;
  help: boolean;
}

/** Parse `analyze` argv. Throws on unknown flags (parseArgs) or invalid option values (zod). */
export function parseCliArgs(argv: string[]): CliInvocation {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      detail: { type: 'string' },
      'max-frames': { type: 'string' },
      'max-width': { type: 'string' },
      fields: { type: 'string' },
      'force-refresh': { type: 'boolean' },
      'ocr-language': { type: 'string' },
      model: { type: 'string' },
      language: { type: 'string' },
      out: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  const raw: Record<string, unknown> = {};
  if (values.detail !== undefined) raw.detail = values.detail;
  if (values['max-frames'] !== undefined) raw.maxFrames = Number(values['max-frames']);
  if (values['max-width'] !== undefined) raw.maxWidth = Number(values['max-width']);
  if (values.fields !== undefined) {
    raw.fields = values.fields
      .split(',')
      .map((f) => f.trim())
      .filter(Boolean);
  }
  if (values['force-refresh']) raw.forceRefresh = true;
  if (values['ocr-language'] !== undefined) raw.ocrLanguage = values['ocr-language'];
  if (values.model !== undefined) raw.model = values.model;
  if (values.language !== undefined) raw.language = values.language;

  // Validation (enum/range) comes from the shared MCP tool schema.
  const options = AnalyzeOptionsSchema.parse(Object.keys(raw).length > 0 ? raw : undefined);

  return {
    url: positionals[0],
    options,
    outDir: values.out,
    help: values.help ?? false,
  };
}

/** Stable per-source frames dir so repeat runs reuse the same folder. */
export function defaultOutDir(url: string): string {
  const hash = createHash('sha256').update(url).digest('hex').slice(0, 12);
  return persistentCacheDir(hash);
}

/**
 * Copy frame images out of the per-call temp dir (about to be cleaned up) into
 * `outDir`, rewriting each `filePath`. Renamed to `slide-NN.<ext>` by ordinal —
 * never derived from `time` values, which contain `:` (illegal on Windows).
 *
 * ENOENT on the source (frame already cleaned up after a cache hit) is the
 * benign case, counted in `missing`. Any other failure (EACCES/ENOSPC/EROFS on
 * the destination) is a real write problem — reported per-frame in `errors`
 * with the actual errno message, never disguised as a cache race.
 */
export async function copyFrames(
  frames: IFrameResult[],
  outDir: string,
): Promise<{ frames: IFrameResult[]; missing: number; errors: string[] }> {
  await mkdir(outDir, { recursive: true });
  const copied: IFrameResult[] = [];
  let missing = 0;
  const errors: string[] = [];
  for (const [i, frame] of frames.entries()) {
    // Named by slide ordinal rather than the temp basename, so `slides/` reads
    // as a deck. Still never derived from `time` — those contain `:`, which is
    // illegal in a Windows filename; a zero-padded index is safe everywhere and
    // keeps the directory in playback order under a plain lexical sort.
    const dest = join(
      outDir,
      `slide-${String(i + 1).padStart(2, '0')}${extname(frame.filePath) || '.jpg'}`,
    );
    try {
      await copyFile(frame.filePath, dest);
      copied.push({ ...frame, filePath: dest });
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        missing++;
      } else {
        errors.push(
          `Frame copy to ${dest} failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }
  }
  return { frames: copied, missing, errors };
}

function formatError(err: unknown): string {
  if (err instanceof ZodError) {
    return err.issues
      .map((issue) => `Invalid option "${String(issue.path[0] ?? '')}": ${issue.message}`)
      .join('\n');
  }
  return err instanceof Error ? err.message : String(err);
}

/**
 * `mcp-video-analyzer analyze` entry point. stdout is reserved for the single
 * JSON result document — everything else (progress, errors) goes to stderr.
 */
/**
 * Copy the audio out of the per-call temp dir and write the standalone
 * transcript artifacts into `outDir`.
 *
 * Runs BEFORE `handle.cleanup()` for the same reason frame copying does: the
 * WAV and MP3 live in the temp dir and are gone the moment it is reclaimed.
 * Every failure is collected as a warning rather than thrown — a document that
 * already has a transcript must not be lost because a file copy failed.
 */
export async function writeAnalysisArtifacts(
  result: IAnalysisResult,
  outDir: string,
  slideFrames: IFrameResult[] = [],
): Promise<{ paths: Record<string, string>; warnings: string[] }> {
  const paths: Record<string, string> = {};
  const warnings: string[] = [];

  const hasAudio = Boolean(result.audio?.wavPath ?? result.audio?.mp3Path);
  const hasTranscript = result.transcript.length > 0;
  const slides = result.slides ?? [];
  // Create `--out` only when something actually lands in it. A run that emits
  // no frames and produced no audio, transcript or slides (a `--fields metadata`
  // query, a silent black clip) must leave the filesystem untouched rather than
  // scattering empty directories — asserted by the CLI smoke test.
  if (!hasAudio && !hasTranscript && slides.length === 0) return { paths, warnings };

  await mkdir(outDir, { recursive: true });

  // `extracted_audio.*` rather than `audio.*`: the name says where it came from,
  // which matters in an --out dir that also holds slides and transcripts.
  const audioCopies: [string | undefined, string, string][] = [
    [result.audio?.wavPath, 'extracted_audio.wav', 'wav'],
    [result.audio?.mp3Path, 'extracted_audio.mp3', 'mp3'],
  ];
  for (const [src, name, key] of audioCopies) {
    if (!src) continue;
    const dest = join(outDir, name);
    try {
      await copyFile(src, dest);
      paths[key] = dest;
    } catch (e: unknown) {
      warnings.push(`Could not keep ${name}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  // Frame images were copied into `slides/` by copyFrames; pair each slide with
  // its image by timestamp so every artifact can point at the picture.
  const frameByTime = new Map(slideFrames.map((f) => [f.time, f.filePath]));
  const enrichedSlides = slides.map((slide) => ({
    ...slide,
    framePath: frameByTime.get(slide.time),
  }));

  if (hasTranscript) {
    const doc = {
      source: result.metadata.url,
      durationSeconds: result.metadata.duration,
      segments: result.transcript.map((t) => ({
        start: parseTimeToSeconds(t.time),
        end: t.endTime === undefined ? undefined : parseTimeToSeconds(t.endTime),
        startTime: t.time,
        endTime: t.endTime,
        text: t.text,
      })),
    };
    try {
      const txtPath = join(outDir, 'transcript.txt');
      await writeFile(txtPath, `${renderTranscriptText(result.transcript)}\n`, 'utf8');
      paths.transcriptTxt = txtPath;
      const jsonPath = join(outDir, 'transcript.json');
      await writeFile(jsonPath, `${JSON.stringify(doc, null, 2)}\n`, 'utf8');
      paths.transcriptJson = jsonPath;
    } catch (e: unknown) {
      warnings.push(
        `Could not write transcript artifacts: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }

  if (enrichedSlides.length > 0) {
    const slidesDir = join(outDir, SLIDES_DIRNAME);
    try {
      await mkdir(slidesDir, { recursive: true });
      const slidesJson = join(slidesDir, 'slides.json');
      await writeFile(
        slidesJson,
        `${JSON.stringify(
          {
            source: result.metadata.url,
            durationSeconds: result.metadata.duration,
            slides: enrichedSlides,
          },
          null,
          2,
        )}\n`,
        'utf8',
      );
      paths.slidesDir = slidesDir;
      paths.slidesJson = slidesJson;
    } catch (e: unknown) {
      warnings.push(
        `Could not write slides artifacts: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }

  // The final synchronized timeline, in both machine and human form: `slides`
  // is the slide-windowed view (what was said about each slide) and `events` is
  // the raw chronological merge of transcript, frame and OCR rows.
  if (enrichedSlides.length > 0 || result.timeline.length > 0) {
    try {
      const timelineJson = join(outDir, 'timeline.json');
      await writeFile(
        timelineJson,
        `${JSON.stringify(
          {
            source: result.metadata.url,
            durationSeconds: result.metadata.duration,
            slides: enrichedSlides,
            events: result.timeline,
          },
          null,
          2,
        )}\n`,
        'utf8',
      );
      paths.timelineJson = timelineJson;

      const timelineMd = join(outDir, 'timeline.md');
      await writeFile(timelineMd, renderTimelineMarkdown(result, enrichedSlides), 'utf8');
      paths.timelineMd = timelineMd;
    } catch (e: unknown) {
      warnings.push(
        `Could not write the synchronized timeline: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }

  return { paths, warnings };
}

/** Human-readable synchronized timeline: one section per slide, narration under it. */
function renderTimelineMarkdown(
  result: IAnalysisResult,
  slides: (ISlideNarration & { framePath?: string })[],
): string {
  const title = result.metadata.title || result.metadata.url;
  const spoken = slides.reduce((n, s) => n + s.transcriptEntries, 0);

  const lines: string[] = [
    `# ${title} — synchronized timeline`,
    '',
    `- Duration: ${result.metadata.durationFormatted ?? `${result.metadata.duration ?? 0}s`}`,
    `- Slides: ${slides.length}`,
    `- Transcript segments: ${result.transcript.length} (${spoken} placed on a slide)`,
    '',
  ];

  if (result.transcript.length === 0) {
    lines.push(
      '> No transcript was produced, so the narration sections below are empty.',
      '> The slide text comes from OCR and is unaffected.',
      '',
    );
  }

  for (const slide of slides) {
    lines.push(`## [${slide.time} – ${slide.endTime}] Slide ${slide.slideIndex + 1}`);
    lines.push('');
    if (slide.framePath) lines.push(`![slide ${slide.slideIndex + 1}](${slide.framePath})`, '');
    lines.push('**On screen**', '', '```', slide.ocrText.trim(), '```', '');
    lines.push('**Narration**', '', slide.narration.trim() || '_(none captured)_', '');
  }

  return `${lines.join('\n')}\n`;
}

export async function runCli(argv: string[]): Promise<number> {
  let invocation: CliInvocation;
  try {
    invocation = parseCliArgs(argv);
  } catch (err) {
    process.stderr.write(`${formatError(err)}\n\n${CLI_USAGE}`);
    return 1;
  }

  if (invocation.help) {
    process.stdout.write(CLI_USAGE);
    return 0;
  }

  // A CLI runs from a known shell cwd, so resolve relative local paths before
  // the gate (the MCP server rejects them — its cwd is unpredictable).
  let url = invocation.url;
  if (url && !isAbsolute(url) && !url.includes('://') && existsSync(url)) {
    url = resolve(url);
  }
  if (!url || !isVideoSource(url)) {
    process.stderr.write(
      'Must be a supported video URL (Loom, YouTube, Vimeo, TikTok, Instagram, X/Twitter, Twitch, Dailymotion, Facebook), a direct .mp4/.webm/.mov URL, or a path / file:// URI to a local video file\n',
    );
    return 1;
  }

  registerAllAdapters();

  const progress: ProgressReporter = async (percent, message) => {
    process.stderr.write(`[${Math.round(percent)}%] ${message ?? ''}\n`);
  };

  let handle;
  try {
    handle = await getAnalysis(url, resolveAnalyzeParams(invocation.options), progress);
  } catch (err) {
    process.stderr.write(`${formatError(err)}\n`);
    return 1;
  }

  const { result } = handle;
  const fields = invocation.options?.fields;
  const wantFrames = !fields || fields.includes('frames');

  const outDir = invocation.outDir ?? defaultOutDir(url);
  let frames: IFrameResult[] = [];
  let missing = 0;
  let artifacts: Record<string, string> = {};
  const copyWarnings: string[] = [];
  try {
    if (wantFrames && result.frames.length > 0) {
      const copied = await copyFrames(result.frames, join(outDir, SLIDES_DIRNAME));
      frames = copied.frames;
      missing = copied.missing;
      copyWarnings.push(...copied.errors);
    }
    const written = await writeAnalysisArtifacts(result, outDir, frames);
    artifacts = written.paths;
    copyWarnings.push(...written.warnings);
  } catch (err) {
    // A failed mkdir/copy (bad --out, permissions) must not discard an
    // analysis that already succeeded — degrade into warnings[] and emit the
    // document without frame files (graceful-degradation convention).
    copyWarnings.push(
      `Frame images could not be copied to the output dir: ${err instanceof Error ? err.message : String(err)}`,
    );
  } finally {
    // Always reclaim the per-call temp dir, even when the copy failed.
    await handle.cleanup();
  }

  const doc = assembleResultDoc(result, fields, {
    missingFrames: missing,
    refreshHint: '--force-refresh',
    extraWarnings: copyWarnings,
  });
  if (wantFrames) doc.frames = frames;
  // Where the kept audio and the standalone transcript files ended up, so a
  // caller parsing stdout never has to guess at --out's layout.
  if (Object.keys(artifacts).length > 0) doc.artifacts = artifacts;

  process.stdout.write(`${JSON.stringify(doc, null, 2)}\n`);
  return 0;
}
