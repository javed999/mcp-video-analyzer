import { describe, expect, it } from 'vitest';
import type { IOcrEntry, ITranscriptEntry } from '../types.js';
import { buildSlideNarration, renderTranscriptText } from './slide-sync.js';

const slides: IOcrEntry[] = [
  { time: '0:05', text: 'Case 15: Infectious Diseases', confidence: 94 },
  { time: '0:34', text: 'Physical Examination', confidence: 92 },
  { time: '16:29', text: 'Final Diagnosis: Pyelonephritis', confidence: 94 },
];

const transcript: ITranscriptEntry[] = [
  { time: '0:00', endTime: '0:04', text: 'Welcome back.' }, // before slide 0
  { time: '0:10', endTime: '0:20', text: 'A 25 year old woman.' }, // slide 0
  { time: '0:40', endTime: '0:50', text: 'Temperature 39.2 degrees.' }, // slide 1
  { time: '16:40', endTime: '16:50', text: 'Pyelonephritis.' }, // after last slide start
];

describe('buildSlideNarration', () => {
  it('attaches narration to the slide that was on screen when it was said', () => {
    const result = buildSlideNarration(transcript, slides, 994.93);

    expect(result.map((s) => s.narration)).toEqual([
      // The first window reaches back to 0, so the intro spoken over the title
      // card is kept rather than dropped.
      'Welcome back. A 25 year old woman.',
      'Temperature 39.2 degrees.',
      'Pyelonephritis.',
    ]);
  });

  it('builds half-open windows so no second belongs to two slides', () => {
    const result = buildSlideNarration(transcript, slides, 994.93);

    expect(result[0]).toMatchObject({ slideIndex: 0, seconds: 5, endSeconds: 34 });
    expect(result[1]).toMatchObject({ slideIndex: 1, seconds: 34, endSeconds: 989 });
    // Last window runs to the video duration, absorbing the closing remarks.
    expect(result[2]).toMatchObject({ slideIndex: 2, seconds: 989, endSeconds: 994.93 });
  });

  it('loses no speech: every transcript entry lands on exactly one slide', () => {
    const result = buildSlideNarration(transcript, slides, 994.93);

    // This is the property the whole feature exists to guarantee — a boundary
    // bug that silently swallowed narration would still produce plausible
    // output, so assert the totals rather than eyeballing the windows.
    const assigned = result.reduce((n, s) => n + s.transcriptEntries, 0);
    expect(assigned).toBe(transcript.length);

    for (const entry of transcript) {
      const owners = result.filter((s) => s.narration.includes(entry.text));
      expect(owners).toHaveLength(1);
    }
  });

  it('still terminates the last window when the duration is unknown', () => {
    const result = buildSlideNarration(transcript, slides);

    expect(result).toHaveLength(3);
    expect(result[2].narration).toBe('Pyelonephritis.');
    expect(Number.isFinite(result[2].endSeconds)).toBe(true);
  });

  it('sorts unsorted slide input instead of building overlapping windows', () => {
    const shuffled = [slides[2], slides[0], slides[1]];
    const result = buildSlideNarration(transcript, shuffled, 994.93);

    expect(result.map((s) => s.seconds)).toEqual([5, 34, 989]);
  });

  it('returns [] when there are no slides to key on', () => {
    expect(buildSlideNarration(transcript, [], 994.93)).toEqual([]);
  });

  it('keeps slides that were narrated in silence', () => {
    const result = buildSlideNarration([], slides, 994.93);

    expect(result).toHaveLength(3);
    expect(result.every((s) => s.narration === '' && s.transcriptEntries === 0)).toBe(true);
    // The slide text survives even with no audio — this is the OCR-only case.
    expect(result[0].ocrText).toBe('Case 15: Infectious Diseases');
  });
});

describe('renderTranscriptText', () => {
  it('emits one timestamped line per entry', () => {
    expect(renderTranscriptText(transcript.slice(0, 2))).toBe(
      '[0:00] Welcome back.\n[0:10] A 25 year old woman.',
    );
  });
});
