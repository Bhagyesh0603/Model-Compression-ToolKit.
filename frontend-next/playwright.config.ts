import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/ui',
  timeout: 60_000,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: process.env.BASE_URL ?? 'http://localhost:3100',
    ...devices['Desktop Chrome'],
  },
  webServer: {
    url: 'http://localhost:3100',
    reuseExistingServer: true,
    command: 'npm run dev -- -p 3100',
    timeout: 120_000,
  },
});
