import type { Metadata } from 'next';
import Link from 'next/link';
import { Mark } from '@/src/landing/Mark';

export const metadata: Metadata = { title: 'Page not found', robots: { index: false } };

export default function NotFound() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-md flex-col items-center justify-center px-6 text-center">
      <Mark className="h-10 w-10 text-ink-950" />
      <h1 className="mt-8 text-[clamp(28px,4vw,40px)] font-medium leading-[1.1] tracking-[-0.03em] text-ink-950">
        That page does not exist
      </h1>
      <p className="mt-3 text-[15px] leading-7 text-ink-600">The link may be out of date or mistyped.</p>
      <div className="mt-8 flex flex-col gap-3 sm:flex-row">
        <Link
          href="/"
          className="inline-flex h-11 items-center justify-center rounded-full bg-ink-950 px-6 text-[14px] font-semibold text-white transition-colors hover:bg-ink-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
        >
          Back to compresskit
        </Link>
      </div>
    </main>
  );
}
