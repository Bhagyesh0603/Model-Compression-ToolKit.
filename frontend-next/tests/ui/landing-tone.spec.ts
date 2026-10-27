import { test, expect, type Page } from '@playwright/test';
import { bandBox, contrastOffenders, near, nextFrames, openLanding, rootRgb, scrollForLine, scrollToY, waitToneSettled } from './helpers/landing';

export const BANDS = [
  { id: 'hero', tone: 'light' },
  { id: 'pipeline', tone: 'dark' },
  { id: 'models', tone: 'light' },
  { id: 'limits', tone: 'dark' },
  { id: 'closing', tone: 'light' },
  // The footer is a dark panel that owns its colours; for the page mix it is a light band.
  { id: 'site-footer', tone: 'light' },
] as const;

const LIGHT = [255, 255, 255] as const;
const DARK = [20, 20, 20] as const;

async function recordToneChanges(page: Page): Promise<void> {
  await page.evaluate(() => {
    const w = window as unknown as { __toneChanges: string[] };
    w.__toneChanges = [];
    new MutationObserver((records) => {
      for (const r of records) w.__toneChanges.push((r.target as HTMLElement).dataset.tone ?? '');
    }).observe(document.querySelector('.landing')!, { attributes: true, attributeFilter: ['data-tone'] });
  });
}

const toneChanges = (page: Page) => page.evaluate(() => (window as unknown as { __toneChanges: string[] }).__toneChanges);

test.describe('landing tone bands', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test('each band sets the page tone once the switch line is inside it', async ({ page }) => {
    await openLanding(page);
    for (const band of BANDS) {
      const box = await bandBox(page, band.id);
      await scrollToY(page, scrollForLine(box, box.top + Math.min(box.height / 2, 600)));
      await waitToneSettled(page);
      await expect(page.locator('.landing')).toHaveAttribute('data-tone', band.tone);
      expect(near(await rootRgb(page), band.tone === 'dark' ? DARK : LIGHT), band.id).toBe(true);
    }
  });

  test('scrolling down visits the bands in order without flapping', async ({ page }) => {
    await openLanding(page);
    await recordToneChanges(page);
    const total = await page.evaluate(() => document.documentElement.scrollHeight - window.innerHeight);
    for (let y = 0; y <= total; y += 120) await scrollToY(page, y);
    const changes = await toneChanges(page);
    // light (hero) -> dark -> light -> dark -> light; the page may end on the light footer band.
    expect(changes.slice(0, 4)).toEqual(['dark', 'light', 'dark', 'light']);
  });

  for (const id of ['hero', 'models', 'limits', 'closing']) {
    test(`text in #${id} meets WCAG AA contrast`, async ({ page }) => {
      await openLanding(page);
      const box = await bandBox(page, id);
      await scrollToY(page, scrollForLine(box, box.top + Math.min(box.height / 2, 500)));
      await waitToneSettled(page);
      await nextFrames(page, 3);
      expect(await contrastOffenders(page, id)).toEqual([]);
    });
  }
});

test.describe('nav', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test('marks the Pipeline link while the pipeline band is active, and the current page elsewhere', async ({ page }) => {
    await openLanding(page);
    const box = await bandBox(page, 'pipeline');
    await scrollToY(page, scrollForLine(box, box.top + 400));
    await waitToneSettled(page);
    await expect(page.locator('nav[aria-label="Pages"] a[aria-current="true"]')).toHaveText('Pipeline');
    await page.goto('/explore');
    await expect(page.locator('nav[aria-label="Pages"] a[aria-current="true"]')).toHaveText('Explore');
  });
});

test.describe('pipeline story', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test('scrolling advances the step and its numbers; the tabs jump to a step', async ({ page }) => {
    await openLanding(page);
    const box = await bandBox(page, 'pipeline');
    const tabs = page.locator('#pipeline [role="tab"]');
    await scrollToY(page, box.top + 200);
    await expect(tabs.nth(0)).toHaveAttribute('aria-selected', 'true');
    await scrollToY(page, box.top + box.height - 1000);
    await expect(tabs.nth(4)).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#pipeline').getByText('Run the delivered file, not the model in memory.')).toBeVisible();
    await tabs.nth(3).click();
    await expect(tabs.nth(3)).toHaveAttribute('aria-selected', 'true', { timeout: 6000 });
  });
});

test.describe('hero', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test('ends on the recorded final values and replays on request', async ({ page }) => {
    await openLanding(page);
    const stats = page.locator('#hero dl');
    await expect(stats).toContainText('68,786', { timeout: 8000 });
    await expect(stats).toContainText('84.00%');
    await page.getByRole('button', { name: 'Replay' }).click();
    await expect(stats).toContainText('272,474');
    await expect(stats).toContainText('68,786', { timeout: 8000 });
  });
});

test.describe('closing band', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test('invites without overclaiming', async ({ page }) => {
    await openLanding(page);
    const closing = page.locator('#closing');
    await expect(closing.getByRole('heading', { level: 2 })).toHaveText('Run it on your model, then check the file yourself.');
    await expect(closing.getByRole('link', { name: 'How to run it' })).toHaveAttribute('href', '/run');
    await expect(closing.getByRole('link', { name: 'Explore the results' })).toHaveAttribute('href', '/explore');
    await expect(closing.getByText('Tested on ResNet and VGG-style classifiers', { exact: true })).toBeVisible();
    await expect(closing).not.toContainText(/any CNN|every model|seamless|powerful/i);
  });
});
