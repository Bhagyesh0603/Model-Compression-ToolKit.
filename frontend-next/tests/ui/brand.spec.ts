import { test, expect } from '@playwright/test';

const BRAND_600 = 'rgb(29, 122, 92)';

test.describe('landing brand colour', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test.beforeEach(async ({ page }) => {
    // Skip the opening curtain.
    await page.addInitScript(() => {
      try {
        sessionStorage.setItem('compresskit.intro', '1');
      } catch {
        /* ignore */
      }
    });
    await page.goto('/', { waitUntil: 'load' });
    await page.waitForTimeout(800);
  });

  test('the primary call to action arrow badge is brand-600', async ({ page }) => {
    const badge = page.locator('#hero a[href="/explore"]').first().locator(':scope > span').first();
    await expect(badge).toBeVisible();
    await expect(badge).toHaveCSS('background-color', BRAND_600);
  });

  test('no element uses an orange/red/purple colour (hue < 50 or >= 260, saturation > 0.25)', async ({ page }) => {
    const offenders = await page.evaluate(() => {
      const ctx = document.createElement('canvas').getContext('2d', { willReadFrequently: true })!;
      const toRgba = (value: string): [number, number, number, number] | null => {
        if (!value || value === 'none' || value === 'transparent' || value === 'currentcolor') return null;
        ctx.clearRect(0, 0, 1, 1);
        ctx.fillStyle = '#000';
        ctx.fillStyle = value; // handles rgb(), oklab(), color(srgb ...) from color-mix
        ctx.fillRect(0, 0, 1, 1);
        const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data;
        return [r, g, b, a / 255];
      };
      const hsl = (r: number, g: number, b: number) => {
        const rf = r / 255, gf = g / 255, bf = b / 255;
        const max = Math.max(rf, gf, bf), min = Math.min(rf, gf, bf);
        const l = (max + min) / 2, d = max - min;
        if (d === 0) return { h: 0, s: 0 };
        const s = d / (1 - Math.abs(2 * l - 1));
        let h = max === rf ? ((gf - bf) / d) % 6 : max === gf ? (bf - rf) / d + 2 : (rf - gf) / d + 4;
        h = (h * 60 + 360) % 360;
        return { h, s };
      };
      const props: Array<[string, string]> = [
        ['background-color', 'bg'],
        ['color', 'text'],
        ['border-top-color', 'border'],
        ['fill', 'fill'],
        ['stroke', 'stroke'],
        ['stop-color', 'stop'],
      ];
      const out: string[] = [];
      for (const el of Array.from(document.querySelectorAll('body, body *'))) {
        const cs = getComputedStyle(el);
        for (const [prop, label] of props) {
          const rgba = toRgba(cs.getPropertyValue(prop).trim());
          if (!rgba || rgba[3] < 0.02) continue;
          const { h, s } = hsl(rgba[0], rgba[1], rgba[2]);
          if (s > 0.25 && (h < 50 || h >= 260)) {
            const cls = typeof (el as HTMLElement).className === 'string' ? (el as HTMLElement).className : '';
            out.push(`${el.tagName.toLowerCase()}.${cls.split(/\s+/).slice(0, 4).join('.')} ${label}=rgb(${rgba[0]},${rgba[1]},${rgba[2]}) hue=${Math.round(h)} sat=${s.toFixed(2)}`);
          }
        }
      }
      return Array.from(new Set(out)).slice(0, 40);
    });
    expect(offenders, offenders.join('\n')).toEqual([]);
  });
});
