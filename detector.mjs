import { execFile } from 'node:child_process';
import { mkdir, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { env, pipeline } from '@huggingface/transformers';
import ffmpeg from '@ffmpeg-installer/ffmpeg';

env.cacheDir = fileURLToPath(new URL('./.model-cache/', import.meta.url));
const run = promisify(execFile);
const duration = 30;
const sampleRate = 16000;
let visualModel;
let audioModel;

function loadVisual() {
  visualModel ??= pipeline('image-classification', 'onnx-community/ai-image-detect-distilled-ONNX', {
    revision: '7f067e23521eeb6d6525221af82c613fb746aaff', dtype: 'q8',
  }).catch(error => {
    visualModel = undefined;
    throw error;
  });
  return visualModel;
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

function decode(args) {
  return run(ffmpeg.path, ['-nostdin', '-v', 'error', ...args], {
    encoding: 'buffer', maxBuffer: 4 * 1024 * 1024, timeout: 30000, windowsHide: true,
  });
}

function summarize(label, samples) {
  if (!samples.length) return { label, status: 'unavailable', score: 0, scorePercent: 0, realPercent: 0 };
  const peak = samples.reduce((highest, sample) => sample.score > highest.score ? sample : highest);
  const score = peak.score;
  const scorePercent = Math.round(score * 100);
  const realPercent = 100 - scorePercent;
  const status = score >= 0.50 ? 'flagged' : 'low';
  return {
    label,
    status,
    score,
    scorePercent,
    realPercent,
    time: peak.time,
  };
}

async function classify(model, input, time) {
  const scores = await model(input, { top_k: 2 });
  const score = scores.find(item => item.label.toLowerCase() === 'fake')?.score;
  if (!Number.isFinite(score)) throw new Error('Fake class missing from model response');
  return { score, time };
}

async function visuals(path, type, folder) {
  if (type === 'photo') {
    return [await classify(await loadVisual(), path)];
  }

  const frames = join(folder, 'frames');
  await mkdir(frames);
  await decode(['-t', String(duration), '-i', path, '-an', '-vf',
    'fps=1/2:start_time=0:round=up,scale=384:384:force_original_aspect_ratio=decrease',
    '-frames:v', '15', join(frames, '%02d.png')]);
  const names = (await readdir(frames)).sort();
  if (!names.length) return [];
  const model = await loadVisual();
  const samples = [];
  for (const [index, name] of names.entries()) {
    samples.push(await classify(model, join(frames, name), index * 2));
  }
  return samples;
}

async function speech(path) {
  const { stdout } = await decode(['-i', path, '-t', String(duration), '-map', '0:a:0?',
    '-vn', '-ac', '1', '-ar', String(sampleRate), '-f', 'f32le', 'pipe:1']);
  const chunkSize = sampleRate * 5;
  const samples = [];
  for (let start = 0; start + sampleRate <= stdout.length / 4; start += chunkSize) {
    const length = Math.min(chunkSize, stdout.length / 4 - start);
    const input = Float32Array.from({ length }, (_, index) => stdout.readFloatLE((start + index) * 4));
    if (input.every(value => Math.abs(value) < 0.0001)) continue;
    samples.push(await classify(await loadAudio(), input, start / sampleRate));
  }
  return samples;
}

export async function scan(path, type, folder) {
  const results = [];
  async function check(label, runCheck) {
    try {
      results.push(summarize(label, await runCheck()));
    } catch (error) {
      console.error(label + ' check failed:', error.message);
      results.push({ label, status: 'unavailable' });
    }
  }
  if (type !== 'audio') await check('AI visuals', () => visuals(path, type, folder));
  if (type !== 'photo') await check('AI awaaz', () => speech(path));
  return results;
}
