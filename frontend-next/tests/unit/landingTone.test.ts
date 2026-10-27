import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const LANDING = path.join(process.cwd(), 'src/landing');
const TONE_AWARE = ['Hero.tsx', 'PipelineStory.tsx', 'ModelsCompare.tsx', 'Limits.tsx', 'Closing.tsx', 'SiteNav.tsx', 'SiteFooter.tsx'];
const BAND_FILES = ['Closing.tsx', 'SiteFooter.tsx'];

const read = (file: string) => fs.readFileSync(path.join(LANDING, file), 'utf8');
/** Literal colours are allowed only inside tone-fixed regions (art, glass, the 3D scene). */
const stripFixed = (src: string) => src.replace(/tone-fixed:start[\s\S]*?tone-fixed:end/g, '');

const RULES: Array<[RegExp, string]> = [
  [/(?<![\w-])bg-white(?![\w/-])/, 'bg-white → bg-(--bg) or bg-(--inv-bg)'],
  [/\[#(?:141414|6b6b6b|5c5c5c|a3a3a3|efefef|e6e6e6|e4e4e4|2a2a2a)\]/i, 'literal neutral → --fg / --muted / --surface / --line / --inv-*'],
  [/(?<![\w-])(?:text|border|bg)-white\/\d/, 'white alpha → --line-strong / --hover / --chip'],
];

describe('landing tone tokens', () => {
  it('tone-aware sections use tone tokens outside tone-fixed regions', () => {
    const offenders: string[] = [];
    for (const file of TONE_AWARE) {
      stripFixed(read(file))
        .split('\n')
        .forEach((line) => {
          for (const [re, fix] of RULES) if (re.test(line)) offenders.push(`${file}: "${line.trim().slice(0, 90)}" — ${fix}`);
        });
    }
    expect(offenders, offenders.join('\n')).toEqual([]);
  });

  it('every band declares itself and its tone', () => {
    for (const file of BAND_FILES) {
      const src = read(file);
      expect(src, file).toContain('data-tone-band');
      expect(src, file).toMatch(/data-tone="(?:light|dark)"/);
    }
  });

  it('nothing still uses the old data-nav observer', () => {
    const files = fs.readdirSync(LANDING).filter((f) => f.endsWith('.tsx'));
    expect(files.filter((f) => read(f).includes('data-nav='))).toEqual([]);
  });
});
