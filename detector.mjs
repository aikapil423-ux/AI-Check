import { execFile } from 'node:child_process';
import { mkdir, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { env, pipeline, RawImage } from '@huggingface/transformers';
import ffmpeg from '@ffmpeg-installer/ffmpeg';

env.cacheDir = fileURLToPath(new URL('./.model-cache/', import.meta.url));
const run = promisify(execFile);
const duration = 30;
const sampleRate = 16000;
let capModel;
let expModel;
let audioModel;

function loadCapModel() {
  capModel ??= pipeline('image-classification', 'onnx-community/ai-image-detection-ONNX').catch(error => {
    capModel = undefined;
    throw error;
  });
  return capModel;
}

function loadExpModel() {
  expModel ??= pipeline('image-classification', 'prithivMLmods/Deepfake-Detection-Exp-02-22-ONNX').catch(error => {
    expModel = undefined;
    throw error;
  });
  return expModel;
}

function loadAudio() {
  audioModel ??= pipeline('audio-classification', 'as1605/Deepfake-audio-detection-V2', {
    revision: '3aeb18add053e945dc69025147afab0d70fa0188', dtype: 'fp32',
  }).catch(error => {
    audioModel = undefined;
    throw error;
  });
  return audioModel;
}

async function classifyVisual(input, time) {
  const [cap, exp] = await Promise.all([loadCapModel(), loadExpModel()]);
  const [resCap, resExp] = await Promise.all([
    cap(input, { top_k: 2 }),
    exp(input, { top_k: 2 }),
  ]);

  const scores = [readScore(resCap, 'fake'), readScore(resExp, 'deepfake')];
  return { scores, time };
}

function decode(args) {
  return run(ffmpeg.path, ['-nostdin', '-v', 'error', ...args], {
    encoding: 'buffer', maxBuffer: 4 * 1024 * 1024, timeout: 30000, windowsHide: true,
  });
}

async function samplingPlan(path) {
  // FFmpeg reports container duration on stderr when no output is specified.
  const metadata = await run(ffmpeg.path, ['-nostdin', '-hide_banner', '-i', path], {
    timeout: 10000, windowsHide: true,
  }).catch(error => error.stderr ?? '');
  const match = /Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/.exec(metadata);
  const total = match ? Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3]) : null;
  if (total === null || total <= duration) return [{ start: 0, length: total || duration }];

  const length = duration / 3;
  return [0, (total - length) / 2, total - length].map(start => ({ start, length }));
}

function readScore(classes, label) {
  const score = classes.find(item => item.label.toLowerCase() === label)?.score;
  if (!Number.isFinite(score) || score < 0 || score > 1) {
    throw new Error('Invalid classifier score for ' + label);
  }
  return score;
}

async function looksLikeMonochromeGraphic(image) {
  const thumbnail = (await image.resize(64, 64)).rgb();
  const pixels = thumbnail.data;
  let neutral = 0;
  let light = 0;
  let dark = 0;
  for (let i = 0; i < pixels.length; i += 3) {
    const min = Math.min(pixels[i], pixels[i + 1], pixels[i + 2]);
    const max = Math.max(pixels[i], pixels[i + 1], pixels[i + 2]);
    if (max - min <= 18) neutral++;
    if (min >= 225) light++;
    if (max <= 65) dark++;
  }
  const count = pixels.length / 3;
  return neutral >= count * 0.95 && light >= count * 0.45 && dark >= count * 0.04;
}

function summarize(label, samples, type, isVisual) {
  if (!samples.length) return { label, status: 'unavailable', detail: 'Is file se check ke liye usable sample nahi mila.' };

  if (type === 'photo') {
    const [sample] = samples;
    const [imageScore, deepfakeScore] = sample.scores;
    // These screening cutoffs separate the two labeled reference photos.
    const highImageSignal = imageScore >= 0.95;
    const highDeepfakeSignal = deepfakeScore >= 0.9;
    const lowSignal = imageScore <= 0.65 && !highDeepfakeSignal && !sample.monochromeGraphic;
    const status = highImageSignal ? 'flagged' : lowSignal ? 'low' : 'unclear';

    let detail = 'Photo ka AI image score review range mein hai.';
    if (highImageSignal) {
      detail = 'AI image model ka score strong hai. Deepfake score alag dikhaya gaya hai.';
    } else if (highDeepfakeSignal) {
      detail = 'Deepfake model ka score high hai. Photo ko aur verify karna hoga.';
    } else if (sample.monochromeGraphic) {
      detail = 'Image black-and-white logo ya graphic jaisi hai. Model score se AI use ka clear result nahi nikla.';
    } else if (lowSignal) {
      detail = 'Is photo par kisi model ne strong AI signal flag nahi kiya.';
    }

    return {
      label,
      status,
      detail,
      models: ['AI image model', 'Deepfake model'],
      samples: [{ ...sample, signal: highImageSignal ? 'strong' : lowSignal ? 'low' : 'mixed' }],
    };
  }

  // Conservative starting points. Scores are not calibrated probabilities.
  const highThreshold = isVisual ? 0.9 : 0.95;
  const strong = samples.map(sample => sample.scores.every(score => score >= highThreshold));
  const interval = isVisual ? 2 : 5;
  const firstRepeated = strong.findIndex((flagged, index) =>
    flagged && strong[index + 1] && samples[index].window === samples[index + 1].window &&
    Math.abs(samples[index + 1].time - samples[index].time - interval) < 0.01
  );
  const flagged = firstRepeated >= 0;
  const low = samples.every(sample => sample.scores.every(score => score <= 0.2));
  const status = flagged ? 'flagged' : low ? 'low' : 'unclear';

  let detail;
  if (flagged) {
    detail = 'Do lagatar samples mein strong signal mila.';
  } else if (low) {
    detail = 'Check kiye gaye samples mein model scores kam rahe.';
  } else {
    detail = isVisual
      ? 'Models ke scores mixed hain ya signal repeat nahi hua.'
      : 'Awaaz ka signal weak tha ya lagatar chunks mein repeat nahi hua.';
  }

  return {
    label,
    status,
    detail,
    time: flagged ? samples[firstRepeated].time : undefined,
    reviewTimes: samples.filter(sample => sample.scores.some(score => score >= highThreshold))
      .map(sample => sample.time),
    models: isVisual ? ['AI image model', 'Deepfake model'] : ['AI awaaz model'],
    samples: samples.map((sample, index) => ({
      ...sample,
      signal: strong[index] ? 'strong' : sample.scores.every(score => score <= 0.2) ? 'low' : 'mixed',
    })),
  };
}
async function visuals(path, type, folder, windows) {
  if (type === 'photo') {
    const image = await RawImage.read(path);
    const sample = await classifyVisual(image);
    return [{ ...sample, monochromeGraphic: await looksLikeMonochromeGraphic(image) }];
  }

  const samples = [];
  for (const [window, { start, length }] of windows.entries()) {
    const frames = join(folder, 'frames-' + window);
    await mkdir(frames);
    await decode(['-ss', String(start), '-i', path, '-t', String(length), '-an', '-vf',
      'fps=1/2:start_time=0:round=up,scale=384:384:force_original_aspect_ratio=decrease',
      '-frames:v', String(Math.ceil(length / 2)), join(frames, '%02d.png')]);
    const names = (await readdir(frames)).sort();
    for (const [index, name] of names.entries()) {
      const sample = await classifyVisual(join(frames, name), start + index * 2);
      samples.push({ ...sample, window });
    }
  }
  return samples;
}

async function speech(path, windows) {
  const chunkSize = sampleRate * 5;
  const samples = [];
  for (const [window, { start, length }] of windows.entries()) {
    const { stdout } = await decode(['-ss', String(start), '-i', path, '-t', String(length), '-map', '0:a:0?',
      '-vn', '-ac', '1', '-ar', String(sampleRate), '-f', 'f32le', 'pipe:1']);
    for (let offset = 0; offset + sampleRate <= stdout.length / 4; offset += chunkSize) {
      const length = Math.min(chunkSize, stdout.length / 4 - offset);
      const input = Float32Array.from({ length }, (_, index) => stdout.readFloatLE((offset + index) * 4));
      if (input.every(value => Math.abs(value) < 0.0001)) continue;
      const model = await loadAudio();
      const classes = await model(input, { top_k: 2 });
      samples.push({ scores: [readScore(classes, 'fake')], time: start + offset / sampleRate, window });
    }
  }
  return samples;
}
export async function scan(path, type, folder) {
  const results = [];
  const windows = type === 'photo' ? [] : await samplingPlan(path);
  async function check(label, runCheck, isVisual = false) {
    try {
      const result = summarize(label, await runCheck(), type, isVisual);
      results.push({ ...result, ...(windows.length && { windows, sampleKind: isVisual ? 'frame' : 'audio chunk' }) });
    } catch (error) {
      console.error(label + ' check failed:', error.message);
      results.push({ label, status: 'unavailable', detail: 'File ya model process nahi ho saka.' });
    }
  }
  if (type !== 'audio') await check('AI visuals', () => visuals(path, type, folder, windows), true);
  if (type !== 'photo') await check('AI awaaz', () => speech(path, windows));
  return results;
}
