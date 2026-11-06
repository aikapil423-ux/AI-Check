import { createServer } from 'node:http';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { tmpdir } from 'node:os';
import { scan } from './detector.mjs';

const types = {
  photo: ['.jpg', '.jpeg', '.png', '.webp', '.gif'],
  video: ['.mp4', '.mov', '.webm', '.avi', '.mkv', '.wmv'],
  audio: ['.mp3', '.wav', '.m4a', '.flac', '.ogg'],
};
const maxBytes = 20 * 1024 * 1024;
const page = await readFile(new URL('./index.html', import.meta.url));

function json(res, status, data) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(data));
}

createServer(async (req, res) => {
  if (req.method === 'GET' && req.url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    return res.end(page);
  }
  if (req.method !== 'POST' || req.url !== '/api/scan') {
    return json(res, 404, { error: 'Page nahi mili.' });
  }

  try {
    const filename = decodeURIComponent(req.headers['x-file-name'] ?? '');
    const type = Object.keys(types).find(kind => types[kind].includes(extname(filename).toLowerCase()));
    if (!type) return json(res, 400, { error: 'Supported photo, video ya audio select karo.' });
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
    const folder = await mkdtemp(join(tmpdir(), 'ai-check-'));
    try {
      const path = join(folder, 'input' + extname(filename).toLowerCase());
      await writeFile(path, Buffer.concat(parts));
      return json(res, 200, { checks: await scan(path, type, folder) });
    } finally {
      await rm(folder, { recursive: true, force: true });
    }
  } catch {
    return json(res, 400, { error: 'File process nahi ho saki. Dobara try karo.' });
  }
}).listen(Number(process.env.PORT ?? 4173), '127.0.0.1', function () {
  console.log('AI Check: http://127.0.0.1:' + this.address().port);
});
