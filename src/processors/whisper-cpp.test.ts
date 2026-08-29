import { describe, expect, it } from 'vitest';
import {
  buildWhisperCppArgs,
  isDownloadableModel,
  modelDownloadUrl,
  parseWhisperCppJson,
  resolveWhisperCppModel,
} from './whisper-cpp.js';

/**
 * A real whisper.cpp `--output-json` document, matching the writer in
 * `examples/cli/cli.cpp` (`times_o` emits `timestamps` as strings and `offsets`
 * as `t * 10` over whisper's centisecond timebase, i.e. MILLISECONDS).
 *
 * The millisecond scale is the whole reason these timestamps line up with frame
 * and OCR times; reading `offsets` as seconds would put every segment at 0:00
 * and silently destroy the slide synchronisation, so it is pinned here.
 */
const WHISPER_CPP_JSON = JSON.stringify({
  systeminfo: 'AVX2 = 1 |',
  model: { type: 'small', multilingual: false },
  params: { model: 'ggml-small.en.bin', language: 'en' },
  result: { language: 'en' },
  transcription: [
    {
      timestamps: { from: '00:00:00,000', to: '00:00:07,480' },
      offsets: { from: 0, to: 7480 },
      text: ' Case 15, infectious diseases.',
    },
    {
      timestamps: { from: '00:00:07,480', to: '00:00:34,000' },
      offsets: { from: 7480, to: 34000 },
      text: ' The chief complaint is fever, urgency and dysuria.',
    },
    {
      timestamps: { from: '00:16:20,000', to: '00:16:29,120' },
      offsets: { from: 980000, to: 989120 },
      text: ' The final diagnosis is pyelonephritis.',
    },
  ],
});

describe('parseWhisperCppJson', () => {
  it('converts millisecond offsets into clock timestamps that match frame times', () => {
    const entries = parseWhisperCppJson(WHISPER_CPP_JSON);

    expect(entries).toEqual([
      { time: '0:00', endTime: '0:07', text: 'Case 15, infectious diseases.' },
      {
        time: '0:07',
        endTime: '0:34',
        text: 'The chief complaint is fever, urgency and dysuria.',
      },
      // 980000ms = 16m20s — the hour-less `m:ss` form, same as frame times.
      { time: '16:20', endTime: '16:29', text: 'The final diagnosis is pyelonephritis.' },
    ]);
  });

  it('distinguishes "no speech" ([]) from "unusable output" (null)', () => {
    // A run that decoded nothing is a valid, empty transcript.
    expect(parseWhisperCppJson(JSON.stringify({ transcription: [] }))).toEqual([]);

    // Anything that isn't a whisper.cpp document at all is null, so the caller
    // reports a backend failure instead of a silently empty transcript.
    expect(parseWhisperCppJson('not json')).toBeNull();
    expect(parseWhisperCppJson('null')).toBeNull();
    expect(parseWhisperCppJson(JSON.stringify({ result: { language: 'en' } }))).toBeNull();
  });

  it('skips blank and malformed segments without dropping the good ones', () => {
    const entries = parseWhisperCppJson(
      JSON.stringify({
        transcription: [
          { offsets: { from: 0, to: 1000 }, text: '   ' },
          { offsets: { from: 1000, to: 2000 } },
          null,
          { offsets: { from: 2000, to: 3000 }, text: ' kept ' },
        ],
      }),
    );

    expect(entries).toEqual([{ time: '0:02', endTime: '0:03', text: 'kept' }]);
  });

  it('survives a segment with no offsets rather than throwing', () => {
    expect(parseWhisperCppJson(JSON.stringify({ transcription: [{ text: 'hi' }] }))).toEqual([
      { time: '0:00', endTime: undefined, text: 'hi' },
    ]);
  });
});

describe('whisper.cpp model resolution', () => {
  it('defaults to small.en — base.en mishears domain vocabulary', () => {
    expect(resolveWhisperCppModel()).toBe('small.en');
    expect(resolveWhisperCppModel({ model: 'base.en' })).toBe('base.en');
  });

  it('knows which names it can fetch weights for', () => {
    expect(isDownloadableModel('small.en')).toBe(true);
    expect(isDownloadableModel('base.en')).toBe(true);
    // A custom fine-tune is allowed, but must be supplied by the caller.
    expect(isDownloadableModel('my-finetune')).toBe(false);
  });

  it('substitutes the model into the URL template', () => {
    expect(modelDownloadUrl('small.en')).toContain('ggml-small.en.bin');
  });
});

describe('buildWhisperCppArgs', () => {
  it('requests JSON output and pins the model, audio and language', () => {
    const args = buildWhisperCppArgs('/tmp/audio.wav', '/models/ggml-small.en.bin', '/out/base', {
      language: 'en',
    });

    expect(args).toEqual([
      '-m',
      '/models/ggml-small.en.bin',
      '-f',
      '/tmp/audio.wav',
      '-l',
      'en',
      '--output-json',
      '--output-file',
      '/out/base',
    ]);
  });

  it('passes a domain glossary through as --prompt', () => {
    const args = buildWhisperCppArgs('/a.wav', '/m.bin', '/o', {
      initialPrompt: 'pyelonephritis, ceftriaxone',
    });

    expect(args).toContain('--prompt');
    expect(args[args.indexOf('--prompt') + 1]).toBe('pyelonephritis, ceftriaxone');
  });
});
