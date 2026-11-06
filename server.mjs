import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname } from 'node:path';

const types = {
  photo: ['.jpg', '.jpeg', '.png', '.webp', '.gif'],
  video: ['.mp4', '.mov', '.webm', '.avi', '.mkv', '.wmv'],
  audio: ['.mp3', '.wav', '.m4a', '.flac', '.ogg'],
};
const maxBytes = 20 * 1024 * 1024;
const page = await readFile(new URL('./index.html', import.meta.url));

async function hive(file, key) {
  const form = new FormData();
  form.set('media', file);
  const response = await fetch('https://api.thehive.ai/api/v2/task/sync', {
    method: 'POST',
    headers: { Authorization: 'token ' + key },
    body: form,
    signal: AbortSignal.timeout(60000),
  });
  if (!response.ok) throw new Error('Hive request failed');
  const data = await response.json();
  const task = data.status?.[0];
  if (String(task?.status?.code) !== '0' || !Array.isArray(task.response?.output)) {
    throw new Error('Hive returned no results');
  }
  return task.response.output;
}

function summarize(output, name, label, videoFace = false) {
  const samples = output.map(frame => ({
    time: frame.time ?? frame.start_time,
    score: frame.classes?.find(item => item.class === name)?.score,
  }));
  const valid = samples.filter(item => Number.isFinite(item.score) && item.score >= 0 && item.score <= 1);
  if (!valid.length) return { label, status: 'unavailable' };
  const peak = valid.reduce((highest, sample) => sample.score > highest.score ? sample : highest);
  const flagged = videoFace
    ? samples.some((sample, index) => sample.score >= 0.5 && samples[index + 1]?.score >= 0.5)
    : peak.score >= 0.9;
  return {
    label,
    status: flagged ? 'flagged' : peak.score >= 0.5 ? 'unclear' : 'low',
    score: peak.score,
    time: peak.time,
  };
}

async function scan(file, type) {
  const visualKey = process.env.HIVE_VISUAL_KEY;
  const audioKey = process.env.HIVE_AUDIO_KEY;
  const checks = [];
  if (type !== 'audio') {
    checks.push((async () => {
      const output = visualKey ? await hive(file, visualKey).catch(() => []) : [];
      return [
        summarize(output, 'ai_generated', 'AI visuals'),
        summarize(output, 'deepfake', 'Face swap', type === 'video'),
      ];
    })());
  }
  if (type !== 'photo') {
    checks.push((async () => {
      const output = audioKey ? await hive(file, audioKey).catch(() => []) : [];
      return [summarize(output, 'ai_generated', 'AI audio')];
    })());
  }
  return (await Promise.all(checks)).flat();
}

function json(res, status, data) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(data));
}

createServer(async (req, res) => {
  if (req.method === 'GET' && req.url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    return res.end(page);
  }
  if (req.method === 'GET' && req.url === '/api/setup') {
    return json(res, 200, { visual: Boolean(process.env.HIVE_VISUAL_KEY), audio: Boolean(process.env.HIVE_AUDIO_KEY) });
  }
  if (req.method !== 'POST' || req.url !== '/api/scan') {
    return json(res, 404, { error: 'Page nahi mili.' });
  }

  try {
    const filename = decodeURIComponent(req.headers['x-file-name'] ?? '');
    const type = Object.keys(types).find(kind => types[kind].includes(extname(filename).toLowerCase()));
    if (!type) return json(res, 400, { error: 'Supported photo, video ya audio select karo.' });
    const configured = (type !== 'audio' && process.env.HIVE_VISUAL_KEY)
      || (type !== 'photo' && process.env.HIVE_AUDIO_KEY);
    if (!configured) return json(res, 503, { error: 'Is file ke liye Hive API key setup karo.' });

    let size = 0;
    const parts = [];
    for await (const chunk of req.iterator({ destroyOnReturn: false })) {
      size += chunk.length;
      if (size > maxBytes) {
        req.resume();
        return json(res, 413, { error: 'File 20 MB se chhoti honi chahiye.' });
      }
      parts.push(chunk);
    }
    if (!size) return json(res, 400, { error: 'File khaali hai.' });
    const file = new File(parts, filename, { type: req.headers['content-type'] ?? 'application/octet-stream' });
    return json(res, 200, { checks: await scan(file, type) });
  } catch {
    return json(res, 400, { error: 'File process nahi ho saki. Dobara try karo.' });
  }
}).listen(Number(process.env.PORT ?? 4173), '127.0.0.1', function () {
  console.log('AI Check: http://127.0.0.1:' + this.address().port);
});
