/**
 * "No number without a source." Reads every prerendered page and requires each
 * figure that carries a unit (%, x, ms, MB, dB) and each large count to be a
 * value from src/data/*.json, at the precision the page prints it. Derived
 * differences are written as "points" and are deliberately not checked here.
 *
 * Needs a build: `npm run build` first. It skips itself if .next is missing.
 */
import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = path.resolve(__dirname, '../..');
const DATA = path.join(ROOT, 'src/data');
const APP = path.join(ROOT, '.next/server/app');

function numbers(value: unknown, out: number[] = []): number[] {
  if (typeof value === 'number' && Number.isFinite(value)) out.push(value);
  // Figures quoted inside recorded text (a budget limit in a failure message, a size in a notebook cell).
  else if (typeof value === 'string') for (const m of value.matchAll(/\d+(?:\.\d+)?/g)) out.push(Number(m[0]));
  else if (Array.isArray(value)) value.forEach((v) => numbers(v, out));
  else if (value && typeof value === 'object') Object.values(value).forEach((v) => numbers(v, out));
  return out;
}

function allowedSet(): Set<string> {
  const allowed = new Set<string>();
  const add = (n: number) => {
    for (let d = 0; d <= 4; d++) {
      allowed.add(n.toFixed(d));
      allowed.add((n * 100).toFixed(d));
      allowed.add((n / 1e6).toFixed(d));
      allowed.add((n / 1e6 / 1e0).toFixed(d));
    }
  };
  for (const file of fs.readdirSync(DATA).filter((f) => f.endsWith('.json'))) {
    numbers(JSON.parse(fs.readFileSync(path.join(DATA, file), 'utf8'))).forEach(add);
  }
  return allowed;
}

function htmlFiles(dir: string, out: string[] = []): string[] {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) htmlFiles(full, out);
    else if (e.name.endsWith('.html') && !/_not-found|_global-error/.test(e.name)) out.push(full);
  }
  return out;
}

function visibleText(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/g, ' ')
    .replace(/<style[\s\S]*?<\/style>/g, ' ')
    // Chart scaffolding (axis ticks) is computed from the data's range, not a claim.
    .replace(/<svg[\s\S]*?<\/svg>/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&#x27;|&apos;/g, "'")
    .replace(/&nbsp;/g, ' ')
    .replace(/\s+/g, ' ');
}

// A number directly followed by a unit, or a comma-grouped count (parameters).
const TOKEN = /(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?=\s?(?:%|x\b|ms\b|MB\b|dB\b))|(?<![\w.])(\d{1,3}(?:,\d{3})+)(?![\d,])/g;

describe.skipIf(!fs.existsSync(APP))('every printed figure is in the recorded data', () => {
  const allowed = allowedSet();
  const files = fs.existsSync(APP) ? htmlFiles(APP) : [];

  it('finds the built pages', () => {
    expect(files.length).toBeGreaterThan(10);
  });

  for (const file of files) {
    const route = '/' + path.relative(APP, file).replace(/\.html$/, '');
    it(route, () => {
      const text = visibleText(fs.readFileSync(file, 'utf8'));
      const unknown: string[] = [];
      for (const m of text.matchAll(TOKEN)) {
        const raw = (m[1] ?? m[2]).replace(/,/g, '');
        if (!allowed.has(raw)) unknown.push(`${m[0]}  ...${text.slice(Math.max(0, m.index! - 30), m.index! + m[0].length + 8)}`);
      }
      expect(unknown, `figures on ${route} that are not in src/data`).toEqual([]);
    });
  }
});

describe('the audit itself', () => {
  it('rejects a figure that is not in the data', () => {
    expect(allowedSet().has('12.345')).toBe(false);
    expect(allowedSet().has('91.54')).toBe(true);
  });
});
