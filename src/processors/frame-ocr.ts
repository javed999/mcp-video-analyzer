import { mkdir, rm } from 'node:fs/promises';
import type { IFrameResult, IOcrEntry } from '../types.js';
import { persistentCacheDir } from '../utils/temp-files.js';
import { preprocessForOcr } from './image-optimizer.js';

/**
 * OCR result for a single frame. Alias of the canonical {@link IOcrEntry} in
 * types.ts — kept as a named export for the OCR module's API while remaining a
 * single source of truth for the shape.
 */
export type IOcrResult = IOcrEntry;

/** Derive the temp path for a frame's OCR-preprocessed copy (lossless PNG). */
function ocrPreprocessPath(framePath: string): string {
  return `${framePath.replace(/\.[^.\\/]+$/, '')}.ocr.png`;
}

/** A result is "meaningful" (kept in output / trusted for dedup) above these. */
const MIN_TEXT_LENGTH = 3;
const MIN_CONFIDENCE = 50;

/** True for OCR results worth surfacing — filters short/low-confidence noise. */
export function isMeaningfulOcr(result: IOcrResult): boolean {
  return result.text.length > MIN_TEXT_LENGTH && result.confidence > MIN_CONFIDENCE;
}

/**
 * Run OCR on every frame, returning one result per input frame (aligned 1:1 by
 * index). Frames that yield no text or fail recognition come back with an empty
 * string and confidence 0 rather than being dropped — callers that need the
 * raw, per-frame signal (e.g. text-aware dedup) rely on this alignment.
 *
 * Returns `[]` when tesseract.js isn't available (alignment is then the caller's
 * responsibility to detect via length mismatch).
 */
export async function ocrFrames(
  frames: IFrameResult[],
  language = 'eng+por',
  onProgress?: (completed: number, total: number) => void,
): Promise<IOcrResult[]> {
  const Tesseract = await loadTesseract();
  if (!Tesseract) return [];

  // Preprocessing (grayscale + 2× upscale + contrast normalization) materially
  // improves OCR of stylized on-screen text. On by default; set
  // MCP_OCR_PREPROCESS=0 to OCR the raw frames instead.
  const preprocess = process.env.MCP_OCR_PREPROCESS !== '0';

  // Cache ~MB-sized .traineddata downloads in a stable temp dir — tesseract.js
  // defaults to the process cwd, which pollutes whatever directory the
  // server/CLI happens to run from (an agent's project root under npx).
  // A mkdir failure propagates: both callers catch it into an "OCR failed:"
  // warning, which beats a far-away traineddata write error (or a silent
  // fallback to cwd — the very bug this cachePath exists to fix).
  const cachePath = persistentCacheDir('tessdata');
  await mkdir(cachePath, { recursive: true });
  const worker = await createWorkerOrThrow(Tesseract, language, cachePath);

  try {
    const results: IOcrResult[] = [];

    for (let i = 0; i < frames.length; i++) {
      const frame = frames[i];

      let target = frame.filePath;
      let scratch: string | null = null;
      if (preprocess) {
        const out = ocrPreprocessPath(frame.filePath);
        const ok = await preprocessForOcr(frame.filePath, out)
          .then(() => true)
          .catch(() => false);
        if (ok) {
          target = out;
          scratch = out;
        }
      }

      let text = '';
      let confidence = 0;
      try {
        const { data } = await worker.recognize(target);
        text = data.text.trim();
        confidence = Math.round(data.confidence);
      } catch {
        // Recognition failed for this frame — keep the aligned empty entry.
      } finally {
        if (scratch) await rm(scratch, { force: true }).catch(() => undefined);
      }

      results.push({ time: frame.time, text, confidence });
      onProgress?.(i + 1, frames.length);
    }

    return results;
  } finally {
    await worker.terminate();
  }
}

/**
 * Extract text from video frames using OCR (tesseract.js).
 * Useful for screencasts, code demos, error messages, and UI text.
 *
 * Only includes results with meaningful text (confidence > 50%, text length > 3).
 */
export async function extractTextFromFrames(
  frames: IFrameResult[],
  language = 'eng+por',
  onProgress?: (completed: number, total: number) => void,
): Promise<IOcrResult[]> {
  const all = await ocrFrames(frames, language, onProgress);
  return all.filter(isMeaningfulOcr);
}

/**
 * `Tesseract.createWorker` with worker-side failures turned into an ordinary
 * rejection — the whole reason this wrapper exists.
 *
 * tesseract.js reports a worker-side failure (most commonly a `.traineddata`
 * fetch that 403s behind a proxy, or any offline/air-gapped run) twice: once by
 * rejecting the pending action promise, and once out-of-band. The out-of-band
 * half is the dangerous one — with no `errorHandler` option it does a bare
 * `throw` inside the worker's `message` listener, which Node re-raises via
 * `process.nextTick` as an UNCAUGHT EXCEPTION. No `try`/`catch` or `.catch()`
 * at any call site can intercept that, so it killed the entire process and
 * discarded every partial result already in hand (metadata, transcript, frames)
 * — exactly the graceful degradation the callers' "OCR failed:" warning is
 * supposed to provide.
 *
 * Passing `errorHandler` suppresses the throw, but on its own it would trade a
 * crash for a hang: tesseract.js only rejects the `createWorker` promise when
 * the failing action is `load`, so a `loadLanguage` failure (the traineddata
 * case) leaves it pending forever. So we race the handler against creation and
 * reject ourselves, and the rejection lands in the callers' existing catch.
 */
async function createWorkerOrThrow(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  Tesseract: any,
  language: string,
  cachePath: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): Promise<any> {
  let signalFailure: (e: Error) => void = () => undefined;
  const failed = new Promise<never>((_, reject) => {
    signalFailure = reject;
  });
  // Errors keep arriving through `errorHandler` after creation succeeds (a
  // per-frame `recognize` failure reports on the same channel, and is already
  // handled by that call's own rejection). Attaching a handler up front keeps
  // such a late signal from surfacing as an unhandled rejection once nothing is
  // racing this promise any more.
  void failed.catch(() => undefined);

  const creation = Tesseract.createWorker(language, undefined, {
    cachePath,
    errorHandler: (e: unknown) => signalFailure(e instanceof Error ? e : new Error(String(e))),
  });

  // If creation completes after we've already given up on it, terminate the
  // worker rather than leaving the thread running. This can't cover the
  // `loadLanguage` failure itself — there the promise never settles, so the
  // handle needed to terminate is never handed over and the orphaned
  // MessagePort keeps the event loop alive; that residue is why `index.ts`
  // exits the one-shot CLI explicitly instead of waiting for a natural drain.
  let raceSettled = false;
  void creation
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    .then((w: any) => {
      if (raceSettled) void w?.terminate?.();
    })
    .catch(() => undefined);

  try {
    return await Promise.race([creation, failed]);
  } finally {
    raceSettled = true;
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function loadTesseract(): Promise<any> {
  try {
    return await import('tesseract.js');
  } catch {
    return null;
  }
}
