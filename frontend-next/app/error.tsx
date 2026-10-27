'use client';

import Link from 'next/link';
import { Mark } from '@/src/landing/Mark';

export default function Error({ retry }: { error: Error & { digest?: string }; retry: () => void }) {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-md flex-col items-center justify-center px-6 text-center">
      <Mark className="h-10 w-10 text-ink-950" />
      <h1 className="mt-8 text-[clamp(28px,4vw,40px)] font-medium leading-[1.1] tracking-[-0.03em] text-ink-950">
        Something went wrong
      </h1>
      <p className="mt-3 text-[15px] leading-7 text-ink-600">This page could not be shown.</p>
      <div className="mt-8 flex flex-col gap-3 sm:flex-row">
        <button
          type="button"
          onClick={() => retry()}
          className="inline-flex h-11 cursor-pointer items-center justify-center rounded-full bg-ink-950 px-6 text-[14px] font-semibold text-white transition-colors hover:bg-ink-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
        >
          Try again
        </button>
        <Link
          href="/"
          className="inline-flex h-11 items-center justify-center rounded-full border border-ink-200 bg-white px-6 text-[14px] font-semibold text-ink-950 transition-colors hover:border-ink-300 hover:bg-ink-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
        >
          Back to compresskit
        </Link>
      </div>
    </main>
  );
}
