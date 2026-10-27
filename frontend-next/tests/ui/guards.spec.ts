import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

/**
 * Static guards (no browser) that pin the design decisions:
 * Tailwind v4 syntax, the evergreen brand ramp, and no orange/red/purple
 * outside the semantic danger tokens.
 */

const ROOT = path.resolve(__dirname, '../..');
const TEXT_EXT = new Set(['.ts', '.tsx', '.css', '.js', '.jsx', '.mjs', '.md', '.mdx', '.json', '.svg']);

function walk(dir: string, out: string[] = []): string[] {
  if (!fs.existsSync(dir)) return out;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'node_modules' || entry.name === '.next') continue;
      walk(full, out);
    } else if (TEXT_EXT.has(path.extname(entry.name))) {
      out.push(full);
    }
  }
  return out;
}

const rel = (f: string) => path.relative(ROOT, f);
const appAndSrc = () => [...walk(path.join(ROOT, 'app')), ...walk(path.join(ROOT, 'src'))];
const landingFiles = () => [...walk(path.join(ROOT, 'src/landing')), path.join(ROOT, 'app/page.tsx')].filter((f) => fs.existsSync(f));

function findings(files: string[], re: RegExp, filter?: (m: RegExpMatchArray) => boolean): string[] {
  const out: string[] = [];
  for (const file of files) {
    const lines = fs.readFileSync(file, 'utf8').split('\n');
    lines.forEach((line, i) => {
      for (const m of line.matchAll(re)) {
        if (!filter || filter(m)) out.push(`${rel(file)}:${i + 1}  ${m[0]}`);
      }
    });
  }
  return out;
}

function hexToRgb(hex: string): [number, number, number] | null {
  let h = hex.replace('#', '');
  if (h.length === 3 || h.length === 4) h = h.split('').map((c) => c + c).join('');
  if (h.length !== 6 && h.length !== 8) return null;
  const n = parseInt(h.slice(0, 6), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function hsl([r, g, b]: [number, number, number]) {
  const rf = r / 255, gf = g / 255, bf = b / 255;
  const max = Math.max(rf, gf, bf), min = Math.min(rf, gf, bf);
  const l = (max + min) / 2;
  const d = max - min;
  if (d === 0) return { h: 0, s: 0, l };
  const s = d / (1 - Math.abs(2 * l - 1));
  let h: number;
  if (max === rf) h = ((gf - bf) / d) % 6;
  else if (max === gf) h = (bf - rf) / d + 2;
  else h = (rf - gf) / d + 4;
  h = (h * 60 + 360) % 360;
  return { h, s, l };
}

const isForbiddenHue = (c: { h: number; s: number }) => c.s > 0.2 && (c.h < 50 || c.h >= 260);

test.describe('static design guards', () => {
  test('no Tailwind v3 CSS-variable shorthand (rounded-[--x]) under app/ or src/', () => {
    const bad = findings(appAndSrc(), /[a-z]-\[--[a-z0-9-]+\]/g);
    expect(bad, `Use v4 syntax e.g. rounded-(--radius-md):\n${bad.join('\n')}`).toEqual([]);
  });

  test('landing has no orange/red/purple hex colours (sat > 0.2, hue < 50 or >= 260)', () => {
    const bad = findings(landingFiles(), /(?<![&\w-])#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})(?![0-9a-zA-Z_-])/g, (m) => {
      const rgb = hexToRgb(m[0]);
      return !!rgb && isForbiddenHue(hsl(rgb));
    });
    expect(bad, `Use brand-* tokens instead:\n${bad.join('\n')}`).toEqual([]);
  });

  test('landing has no orange/red/purple rgb()/rgba() colours', () => {
    const bad = findings(landingFiles(), /rgba?\(\s*(\d{1,3})\s*[, ]\s*(\d{1,3})\s*[, ]\s*(\d{1,3})/g, (m) => {
      const rgb: [number, number, number] = [Number(m[1]), Number(m[2]), Number(m[3])];
      return isForbiddenHue(hsl(rgb));
    });
    expect(bad, `Use color-mix(in oklab, var(--color-brand-600) N%, transparent) or neutrals:\n${bad.join('\n')}`).toEqual([]);
  });

  test('globals.css @theme defines the brand ramp and danger tokens exactly', () => {
    const css = fs.readFileSync(path.join(ROOT, 'app/globals.css'), 'utf8');
    // Collect every @theme { ... } block (brace-matched).
    const blocks: string[] = [];
    let from = 0;
    for (;;) {
      const open = css.slice(from).match(/@theme[^{]*\{/);
      if (!open || open.index === undefined) break;
      const start = from + open.index + open[0].length;
      let depth = 1;
      let i = start;
      while (i < css.length && depth > 0) {
        if (css[i] === '{') depth++;
        else if (css[i] === '}') depth--;
        i++;
      }
      blocks.push(css.slice(start, i - 1));
      from = i;
    }
    const theme = blocks.join('\n');
    const expected: Record<string, string> = {
      'brand-50': '#eefaf4',
      'brand-100': '#dcf3e8',
      'brand-200': '#bfe9d6',
      'brand-300': '#7fd1ae',
      'brand-400': '#4fb98d',
      'brand-500': '#2a9670',
      'brand-600': '#1d7a5c',
      'brand-700': '#17624b',
      'brand-800': '#124d3c',
      'brand-900': '#0d3b2e',
      'danger-50': '#fef3f2',
      'danger-100': '#fee4e2',
      'danger-600': '#b42318',
      'danger-700': '#912018',
    };
    const actual: Record<string, string | undefined> = {};
    for (const name of Object.keys(expected)) {
      const mm = theme.match(new RegExp(`--color-${name}\\s*:\\s*(#[0-9a-fA-F]{3,8})\\s*;`));
      actual[name] = mm?.[1].toLowerCase();
    }
    expect(actual).toEqual(expected);
  });

  test('no Tailwind orange/amber/purple/violet/fuchsia/pink/rose/red palette classes under app/ or src/', () => {
    const hues = 'orange|amber|purple|violet|fuchsia|pink|rose|red|magenta';
    const utils = 'bg|text|border|ring|ring-offset|from|via|to|fill|stroke|outline|divide|shadow|decoration|accent|caret|placeholder|border-[xytrblse]|inset-ring';
    const re = new RegExp(`(?<![\\w-])(?:${utils})-(?:${hues})-(?:\\d{2,3}|\\[)`, 'g');
    const bad = findings(appAndSrc(), re);
    const vars = findings(appAndSrc(), new RegExp(`--color-(?:${hues})-\\d`, 'g'));
    expect([...bad, ...vars], `Use brand-*/ink-* or danger-* tokens (errors only):\n${[...bad, ...vars].join('\n')}`).toEqual([]);
  });
});
