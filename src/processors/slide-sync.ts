import type { IOcrEntry, ISlideNarration, ITranscriptEntry } from '../types.js';
import { parseTimeToSeconds } from './annotated-timeline.js';

/**
 * Align spoken narration to the slide that was on screen while it was said.
 *
 * The annotated timeline interleaves transcript rows and slide rows as separate
 * chronological events; this answers the different question a slide deck poses
 * — "what did the presenter say *about this slide*" — by turning each OCR'd
 * slide into a window and collecting the speech inside it.
 *
 * Windows are half-open `[start, nextStart)` so no second belongs to two
 * slides, and two deliberate extensions make the mapping total, i.e. every
 * transcript entry lands on exactly one slide:
 *   - the first window starts at 0, absorbing narration spoken before the first
 *     detected slide (title cards and intros always precede the first cut);
 *   - the last window runs to `durationSeconds` (or unbounded when the duration
 *     is unknown), absorbing the closing remarks.
 * Dropping either would silently lose speech, which is the failure this whole
 * feature exists to prevent.
 */
export function buildSlideNarration(
  transcript: ITranscriptEntry[],
  slides: IOcrEntry[],
  durationSeconds?: number,
): ISlideNarration[] {
  if (slides.length === 0) return [];

  const bounds = slides.map((slide, i) => ({
    slide,
    index: i,
    start: parseTimeToSeconds(slide.time),
  }));

  // Guard against unsorted input: a caller handing us time-keyed OCR results in
  // any other order would otherwise build overlapping windows.
  bounds.sort((a, b) => a.start - b.start);

  return bounds.map((entry, i) => {
    const isLast = i === bounds.length - 1;
    // First window reaches back to 0; every other starts at its own slide.
    const start = i === 0 ? 0 : entry.start;

    // The window used for ASSIGNMENT is unbounded on the last slide, which is
    // deliberately not the same as the window we report. whisper can emit a
    // closing segment timestamped a shade past the container's reported
    // duration (rounding, or a duration that just disagrees with the stream);
    // clamping assignment to `durationSeconds` silently dropped that segment,
    // which is the one outcome this function must never produce.
    const assignEnd = isLast ? Number.POSITIVE_INFINITY : bounds[i + 1].start;
    const reportedEnd = isLast ? (durationSeconds ?? entry.start) : bounds[i + 1].start;

    const spoken = transcript.filter((t) => {
      const at = parseTimeToSeconds(t.time);
      return at >= start && at < assignEnd;
    });

    return {
      slideIndex: i,
      time: entry.slide.time,
      seconds: entry.start,
      endTime: formatClock(reportedEnd),
      endSeconds: reportedEnd,
      ocrText: entry.slide.text,
      narration: spoken
        .map((t) => t.text.trim())
        .filter(Boolean)
        .join(' '),
      transcriptEntries: spoken.length,
    };
  });
}

/** `m:ss` / `h:mm:ss`, matching frame and transcript timestamps. */
function formatClock(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(s / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  const seconds = s % 60;
  const mm = hours > 0 ? String(minutes).padStart(2, '0') : String(minutes);
  return `${hours > 0 ? `${hours}:` : ''}${mm}:${String(seconds).padStart(2, '0')}`;
}

/**
 * Render a transcript as plain text for `transcript.txt`: one `[mm:ss] line`
 * per entry. Timestamps are kept (rather than emitting a wall of prose) so the
 * text file can still be cross-referenced against the slides and the JSON.
 */
export function renderTranscriptText(transcript: ITranscriptEntry[]): string {
  return transcript.map((t) => `[${t.time}] ${t.text.trim()}`).join('\n');
}
