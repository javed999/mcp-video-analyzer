import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { isMeaningfulOcr, ocrFrames } from './frame-ocr.js';

interface MockWorker {
  recognize: (target: string) => Promise<{ data: { text: string; confidence: number } }>;
  terminate: () => Promise<void>;
}
interface MockWorkerOptions {
  cachePath: string;
  errorHandler?: (e: unknown) => void;
}

const createWorker = vi.hoisted(() =>
  vi.fn(
    async (
      _language: string,
      _oem: undefined,
      _options: MockWorkerOptions,
    ): Promise<MockWorker> => ({
      recognize: async () => ({ data: { text: 'mocked text', confidence: 90 } }),
      terminate: async () => undefined,
    }),
  ),
);

vi.mock('tesseract.js', () => ({ createWorker, default: { createWorker } }));

const frame = () => ({
  time: '0:00',
  filePath: join(tmpdir(), 'nonexistent-frame.jpg'),
  mimeType: 'image/jpeg',
});

describe('isMeaningfulOcr', () => {
  it('requires text length > 3 AND confidence > 50 (both strict)', () => {
    expect(isMeaningfulOcr({ time: '0:01', text: 'R$ 99', confidence: 88 })).toBe(true);
    // length exactly 3 → rejected
    expect(isMeaningfulOcr({ time: '0:01', text: 'abc', confidence: 90 })).toBe(false);
    // length 4 → accepted (with high confidence)
    expect(isMeaningfulOcr({ time: '0:01', text: 'abcd', confidence: 90 })).toBe(true);
    // confidence exactly 50 → rejected
    expect(isMeaningfulOcr({ time: '0:01', text: 'abcde', confidence: 50 })).toBe(false);
    // confidence 51 → accepted
    expect(isMeaningfulOcr({ time: '0:01', text: 'abcde', confidence: 51 })).toBe(true);
    // empty text → rejected
    expect(isMeaningfulOcr({ time: '0:01', text: '', confidence: 99 })).toBe(false);
  });
});

describe('ocrFrames', () => {
  it('routes traineddata downloads to the tmp cache dir, never the process cwd', async () => {
    const results = await ocrFrames([frame()], 'eng');

    expect(results).toHaveLength(1);
    expect(createWorker).toHaveBeenCalledWith('eng', undefined, {
      cachePath: join(tmpdir(), 'mcp-video-analyzer', 'tessdata'),
      errorHandler: expect.any(Function),
    });
  });
});

/**
 * Regression guards for the crash where a `.traineddata` fetch failure took the
 * whole process down (analysis JSON never emitted, already-extracted frames and
 * transcript discarded) instead of degrading into the callers' "OCR failed:"
 * warning.
 *
 * The mocks below model `node_modules/tesseract.js/src/createWorker.js`'s
 * `status === 'reject'` branch, which is the code that made this reachable:
 *
 *     promises[promiseId].reject(data);
 *     if (action === 'load') workerResReject(data);
 *     if (errorHandler) { errorHandler(data); } else { throw Error(data); }
 *
 * Two properties of that branch matter, and each has a test here: the bare
 * `throw` runs inside a worker `message` listener (Node re-raises it via
 * `process.nextTick` as an uncaught exception that no call-site `catch` can
 * reach), and `workerResReject` fires for `action === 'load'` ONLY — so a
 * `loadLanguage` failure leaves the `createWorker` promise pending forever.
 */
describe('ocrFrames — worker failures degrade instead of killing the process', () => {
  it('always passes an errorHandler, so tesseract.js never takes its `throw` branch', async () => {
    createWorker.mockClear();
    await ocrFrames([frame()], 'eng');

    const options = createWorker.mock.calls.at(-1)?.[2];
    expect(typeof options?.errorHandler).toBe('function');
  });

  it('rejects when traineddata loading fails, rather than hanging forever', async () => {
    // A `loadLanguage` failure exactly as tesseract.js reports it: the reason
    // arrives ONLY through `errorHandler`, and the returned promise is never
    // settled. Pre-fix (no errorHandler passed) the real library threw
    // out-of-band here and killed the process; with the throw suppressed but
    // creation un-raced, this await would simply never return — so this test
    // fails, by timeout, against any version that doesn't race the handler.
    createWorker.mockImplementationOnce(
      (_language: string, _oem: undefined, options: MockWorkerOptions) =>
        new Promise<MockWorker>(() => {
          setTimeout(() => {
            options.errorHandler?.(
              'Error: Network error while fetching https://cdn.jsdelivr.net/npm/@tesseract.js-data/eng/4.0.0_best_int/eng.traineddata.gz. Response code: 403',
            );
          }, 0);
        }),
    );

    await expect(ocrFrames([frame()], 'eng')).rejects.toThrow(/Response code: 403/);
  });

  it('keeps a post-creation worker error from becoming an unhandled rejection', async () => {
    // `errorHandler` keeps firing after creation succeeds — a per-frame
    // `recognize` failure reports on the same channel, and is already handled
    // by that call's own rejection. Rejecting an unwatched promise for it would
    // just move the process kill from an uncaught exception to an unhandled
    // rejection, so guard the second door too.
    let handler: ((e: unknown) => void) | undefined;
    createWorker.mockImplementationOnce(async (_language, _oem, options: MockWorkerOptions) => {
      handler = options.errorHandler;
      return {
        recognize: async () => {
          throw new Error('recognition failed');
        },
        terminate: async () => undefined,
      };
    });

    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    try {
      // A failed recognize still yields the aligned empty entry, never a drop.
      await expect(ocrFrames([frame()], 'eng')).resolves.toEqual([
        { time: '0:00', text: '', confidence: 0 },
      ]);

      expect(handler).toBeTypeOf('function');
      handler?.('recognition failed');
      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(unhandled).toEqual([]);
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  });
});
