import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

/**
 * Unit tests for pure logic only (tone selection, snake geometry, colour
 * maths, static guards). Browser behaviour lives in Playwright (tests/ui).
 * *.spec.ts files are Playwright's; Vitest only picks up *.test.ts.
 */
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts', 'tests/unit/**/*.test.ts'],
  },
  resolve: {
    alias: [{ find: /^@\//, replacement: fileURLToPath(new URL('./', import.meta.url)) }],
  },
});
