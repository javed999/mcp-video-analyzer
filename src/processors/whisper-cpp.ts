import { execFile as execFileCb } from 'node:child_process';
import { createWriteStream } from 'node:fs';
import { mkdir, rename, rm, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { promisify } from 'node:util';
import type { ITranscriptEntry } from '../types.js';
import { persistentCacheDir } from '../utils/temp-files.js';

const execFile = promisify(execFileCb);

/**
 * Thrown (as a rejection reason) when no whisper.cpp binary can be located.
 * Callers turn this into an actionable install hint rather than a hard failure.
 */
export const WHISPER_CPP_MISSING = 'WHISPER_CPP_MISSING';

/**
 * Default model. `small.en` is the recommended default for lecture/technical
 * audio — `base.en` mishears domain vocabulary (drug names, jargon) often
 * enough to matter, and the size difference (~466MB vs ~142MB) is a one-off
 * download that is cached across runs.
 */
const DEFAULT_MODEL = 'small.en';

/** Models whisper.cpp publishes in ggml form, smallest first. */
const KNOWN_MODELS = new Set([
  'tiny',
  'tiny.en',
  'base',
  'base.en',
  'small',
  'small.en',
  'medium',
  'medium.en',
  'large-v1',
  'large-v2',
  'large-v3',
  'large-v3-turbo',
]);

/**
 * Where ggml weights are fetched from when the model isn't already cached.
 *
 * whisper.cpp publishes its converted ggml weights in exactly one place, so
 * that is the default. It is a plain file download — inference is entirely
 * local, and nothing is ever sent to the host. Deployments that cannot reach
 * it (an air-gapped runner, or a network policy that blocks the host) point
 * `WHISPER_CPP_MODEL_URL` at their own mirror, or skip downloading altogether
 * by pre-placing the file and setting `WHISPER_CPP_MODEL_PATH`.
 *
 * `{model}` is substituted with the resolved model name.
 */
const DEFAULT_MODEL_URL =
  'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin';

/** A ggml model smaller than this is a stub/truncated download, not weights. */
const MIN_MODEL_BYTES = 10 * 1024 * 1024;

/** Binary names to try on PATH, in order. */
const BIN_CANDIDATES = ['whisper-cli', 'whisper.cpp'];

export interface WhisperCppOptions {
  /** Model name (e.g. `small.en`); overrides WHISPER_CPP_MODEL. */
  model?: string;
  /** Forced language code; overrides WHISPER_CPP_LANGUAGE. */
  language?: string;
  /** Domain glossary passed to whisper.cpp as `--prompt`. */
  initialPrompt?: string;
}

/**
 * Resolve the model name to use. An unknown name is passed through untouched —
 * whisper.cpp accepts any ggml file, and a caller pointing at a custom
 * fine-tune shouldn't be second-guessed — but it can't be auto-downloaded.
 */
export function resolveWhisperCppModel(opts: WhisperCppOptions = {}): string {
  return opts.model || process.env.WHISPER_CPP_MODEL || DEFAULT_MODEL;
}

/** True when the name is one we know how to fetch weights for. */
export function isDownloadableModel(model: string): boolean {
  return KNOWN_MODELS.has(model);
}

/** The URL ggml weights for `model` are fetched from. */
export function modelDownloadUrl(model: string): string {
  const template = process.env.WHISPER_CPP_MODEL_URL || DEFAULT_MODEL_URL;
  return template.replace('{model}', model);
}

/**
 * Locate a whisper.cpp CLI binary, or reject with {@link WHISPER_CPP_MISSING}.
 *
 * Order: explicit `WHISPER_CPP_BIN`, then a build tree named by
 * `WHISPER_CPP_DIR`, then the standard names on PATH. Every candidate is
 * verified by actually running `--help`, so a stale path or a non-executable
 * file is treated as absent rather than blowing up mid-transcription.
 */
async function findWhisperCpp(): Promise<string> {
  const candidates: string[] = [];
  if (process.env.WHISPER_CPP_BIN) candidates.push(process.env.WHISPER_CPP_BIN);
  if (process.env.WHISPER_CPP_DIR) {
    candidates.push(join(process.env.WHISPER_CPP_DIR, 'build', 'bin', 'whisper-cli'));
    candidates.push(join(process.env.WHISPER_CPP_DIR, 'whisper-cli'));
  }
  candidates.push(...BIN_CANDIDATES);

  for (const candidate of candidates) {
    try {
      await execFile(candidate, ['--help'], { timeout: 20000 });
      return candidate;
    } catch (e: unknown) {
      // `--help` exits non-zero on some builds; that still proves the binary
      // runs. Only a spawn failure (ENOENT/EACCES) means "not here".
      const code = (e as { code?: unknown }).code;
      if (code !== 'ENOENT' && code !== 'EACCES' && code !== undefined) return candidate;
    }
  }

  return Promise.reject(new Error(WHISPER_CPP_MISSING));
}

/**
 * Ensure ggml weights for `model` exist on disk, downloading them once into a
 * shared cache dir if needed. Returns the model file path.
 *
 * The download is written to a temp name and renamed into place only after the
 * size check passes, so an interrupted or policy-blocked transfer can never
 * leave a half-file that later loads as a broken model.
 */
async function ensureWhisperCppModel(
  model: string,
  onProgress?: (message: string) => void,
): Promise<string> {
  const explicit = process.env.WHISPER_CPP_MODEL_PATH;
  if (explicit) {
    const size = await fileSize(explicit);
    if (size === null) {
      throw new Error(`WHISPER_CPP_MODEL_PATH points at a missing file: ${explicit}`);
    }
    return explicit;
  }

  const dir = persistentCacheDir('whisper-cpp');
  await mkdir(dir, { recursive: true });
  const modelPath = join(dir, `ggml-${model}.bin`);

  const existing = await fileSize(modelPath);
  if (existing !== null && existing >= MIN_MODEL_BYTES) return modelPath;
  if (existing !== null) await rm(modelPath, { force: true });

  if (!isDownloadableModel(model)) {
    throw new Error(
      `No ggml weights cached for "${model}" and it is not a known whisper.cpp model. ` +
        `Place the file at ${modelPath}, or set WHISPER_CPP_MODEL_PATH.`,
    );
  }

  const url = modelDownloadUrl(model);
  onProgress?.(`Downloading whisper.cpp model "${model}" (one-off) …`);

  const tmpPath = `${modelPath}.${process.pid}.part`;
  try {
    const res = await fetch(url, { redirect: 'follow' });
    if (!res.ok || !res.body) {
      throw new Error(`HTTP ${res.status} ${res.statusText}`);
    }
    await pipeline(Readable.fromWeb(res.body), createWriteStream(tmpPath));

    const downloaded = await fileSize(tmpPath);
    if (downloaded === null || downloaded < MIN_MODEL_BYTES) {
      throw new Error(
        `downloaded file is ${downloaded ?? 0} bytes, too small to be ggml weights ` +
          `(a captive portal or proxy error page is the usual cause)`,
      );
    }
    await rename(tmpPath, modelPath);
  } catch (e: unknown) {
    await rm(tmpPath, { force: true }).catch(() => undefined);
    const reason = e instanceof Error ? e.message : String(e);
    throw new Error(
      `Could not download whisper.cpp model "${model}" from ${url}: ${reason}. ` +
        `Fetch it manually to ${modelPath}, or set WHISPER_CPP_MODEL_PATH / WHISPER_CPP_MODEL_URL.`,
      { cause: e },
    );
  }

  onProgress?.(`whisper.cpp model "${model}" ready.`);
  return modelPath;
}

/**
 * Parse whisper.cpp's `--output-json` document into transcript entries.
 *
 * Pure and total: anything that isn't a well-formed whisper.cpp document
 * returns `null` (so the caller can tell "backend produced nothing usable"
 * apart from "audio genuinely had no speech", which is `[]`).
 *
 * `offsets` are milliseconds (whisper.cpp writes `t * 10` over its centisecond
 * timebase), which is what makes these timestamps line up with frame times.
 */
export function parseWhisperCppJson(raw: string): ITranscriptEntry[] | null {
  let doc: unknown;
  try {
    doc = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof doc !== 'object' || doc === null) return null;

  const segments = (doc as { transcription?: unknown }).transcription;
  if (!Array.isArray(segments)) return null;

  const entries: ITranscriptEntry[] = [];
  for (const segment of segments) {
    if (typeof segment !== 'object' || segment === null) continue;
    const { text, offsets } = segment as {
      text?: unknown;
      offsets?: { from?: unknown; to?: unknown };
    };
    if (typeof text !== 'string') continue;
    const trimmed = text.trim();
    if (!trimmed) continue;

    const fromMs = typeof offsets?.from === 'number' ? offsets.from : 0;
    const toMs = typeof offsets?.to === 'number' ? offsets.to : undefined;

    entries.push({
      time: formatClock(fromMs / 1000),
      endTime: toMs === undefined ? undefined : formatClock(toMs / 1000),
      text: trimmed,
    });
  }

  return entries;
}

/** `m:ss` / `h:mm:ss`, matching the format frames and OCR results use. */
function formatClock(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(s / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  const seconds = s % 60;
  const mm = hours > 0 ? String(minutes).padStart(2, '0') : String(minutes);
  return `${hours > 0 ? `${hours}:` : ''}${mm}:${String(seconds).padStart(2, '0')}`;
}

/** Build the whisper.cpp argv. Exported so the flag contract is unit-testable. */
export function buildWhisperCppArgs(
  audioPath: string,
  modelPath: string,
  outputBase: string,
  opts: WhisperCppOptions = {},
): string[] {
  const language = opts.language || process.env.WHISPER_CPP_LANGUAGE || 'en';
  const prompt = opts.initialPrompt || process.env.WHISPER_PROMPT;
  const threads = process.env.WHISPER_CPP_THREADS;

  const args = [
    '-m',
    modelPath,
    '-f',
    audioPath,
    '-l',
    language,
    '--output-json',
    '--output-file',
    outputBase,
  ];
  if (threads) args.push('-t', threads);
  if (prompt) args.push('--prompt', prompt);
  return args;
}

/**
 * Transcribe with a locally-built whisper.cpp. Fully offline once the model is
 * cached: no API key, no cloud service, no data leaves the machine.
 *
 * Rejects with {@link WHISPER_CPP_MISSING} when no binary is installed so the
 * caller can fall through to another backend; every other failure throws with
 * a reason worth surfacing in `warnings[]`.
 */
export async function transcribeWithWhisperCpp(
  audioPath: string,
  outputDir: string,
  opts: WhisperCppOptions = {},
  onProgress?: (message: string) => void,
): Promise<ITranscriptEntry[]> {
  const bin = await findWhisperCpp();
  const model = resolveWhisperCppModel(opts);
  const modelPath = await ensureWhisperCppModel(model, onProgress);

  const outputBase = join(outputDir, 'whisper-cpp');
  const jsonPath = `${outputBase}.json`;
  onProgress?.(`Transcribing with whisper.cpp (${model}) …`);

  try {
    await execFile(bin, buildWhisperCppArgs(audioPath, modelPath, outputBase, opts), {
      // Long-form audio on CPU is slow; a 16-minute lecture on `small.en`
      // takes minutes, so this ceiling is generous by design.
      timeout: Number(process.env.WHISPER_CPP_TIMEOUT ?? 3_600_000),
      maxBuffer: 64 * 1024 * 1024,
    });
  } catch (e: unknown) {
    const reason = e instanceof Error ? e.message : String(e);
    throw new Error(`whisper.cpp failed: ${reason}`, { cause: e });
  }

  const { readFile } = await import('node:fs/promises');
  const raw = await readFile(jsonPath, 'utf8').catch(() => null);
  if (raw === null) {
    throw new Error(`whisper.cpp produced no JSON at ${jsonPath}`);
  }

  const entries = parseWhisperCppJson(raw);
  if (entries === null) {
    throw new Error('whisper.cpp JSON output could not be parsed');
  }
  return entries;
}

/** `stat` size, or null when the path doesn't exist. */
async function fileSize(path: string): Promise<number | null> {
  try {
    return (await stat(path)).size;
  } catch {
    return null;
  }
}
