import { test, expect, type Page } from '@playwright/test';

const ROUTES = [
  '/',
  '/pipeline',
  '/explore',
  '/techniques',
  '/techniques/pruning',
  '/techniques/gradual',
  '/techniques/recalibration',
  '/techniques/distillation',
  '/techniques/quantization',
  '/techniques/sensitivity',
  '/techniques/search',
  '/techniques/benchmarking',
  '/models',
  '/models/resnet20',
  '/models/vgg16-bn',
  '/notebooks',
  '/notebooks/compresskit-end-to-end',
  '/run',
  '/this-route-does-not-exist',
];

const VIEWPORTS = [
  { name: 'desktop', width: 1440, height: 900 },
  { name: 'mobile', width: 390, height: 844 },
];

const IGNORED_CONSOLE = [/Download the React DevTools/i, /\[HMR\]/i, /Fast Refresh/i];

function watch(page: Page) {
  const pageErrors: string[] = [];
  const consoleErrors: string[] = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  page.on('console', (msg) => {
    if (msg.type() !== 'error') return;
    const text = msg.text();
    if (IGNORED_CONSOLE.some((re) => re.test(text))) return;
    // The browser logs every non-2xx response. A 404 for the page itself (the not-found route)
    // and 401/403 from unauthenticated API probes (session refresh) are expected, not defects.
    if (/Failed to load resource.*status of (401|403)/.test(text) && /\/api\//.test(msg.location().url)) return;
    if (/Failed to load resource.*status of 404/.test(text) && msg.location().url === page.url()) return;
    consoleErrors.push(`${text} @ ${msg.location().url}`);
  });
  return { pageErrors, consoleErrors };
}

for (const vp of VIEWPORTS) {
  test.describe(`routes @ ${vp.name} ${vp.width}x${vp.height}`, () => {
    test.use({ viewport: { width: vp.width, height: vp.height } });

    for (const route of ROUTES) {
      test(`${route} renders cleanly`, async ({ page }) => {
        const { pageErrors, consoleErrors } = watch(page);
        const res = await page.goto(route, { waitUntil: 'load' });
        expect(res, 'navigation response').not.toBeNull();
        expect(res!.status(), 'HTTP status').toBeLessThan(500);
        // Let hydration and client effects run so late errors are caught.
        await page.waitForTimeout(1500);

        expect(pageErrors, 'uncaught page errors').toEqual([]);
        expect(consoleErrors, 'console errors').toEqual([]);

        if (vp.width === 390) {
          const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
          expect(scrollWidth, 'horizontal overflow at 390px').toBeLessThanOrEqual(390);
        }
      });
    }
  });
}

test.describe('typography is Manrope', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test('/ body font-family contains Manrope', async ({ page }) => {
    await page.goto('/', { waitUntil: 'load' });
    const family = await page.evaluate(() => getComputedStyle(document.body).fontFamily);
    expect(family).toMatch(/manrope/i);
  });

});
