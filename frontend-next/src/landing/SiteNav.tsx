'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import { ArrowUpRight, GitBranch } from 'lucide-react';
import { Mark } from './Mark';
import { useLandingTone } from './tone/ToneRoot';

const LINKS = [
  { href: '/#pipeline', label: 'Pipeline', match: (p: string) => p === '/pipeline', band: 'pipeline' },
  { href: '/techniques', label: 'Techniques', match: (p: string) => p.startsWith('/techniques') },
  { href: '/explore', label: 'Explore', match: (p: string) => p.startsWith('/explore') },
  { href: '/models', label: 'Models', match: (p: string) => p.startsWith('/models') },
  { href: '/notebooks', label: 'Notebooks', match: (p: string) => p.startsWith('/notebooks') },
] as const;

/**
 * Fixed top bar. Colours come from the landing tone tokens, so the bar follows
 * the page tone frame by frame; the active-section dot comes from the same
 * band selection (switch line + hysteresis) as the tone, so they never disagree.
 */
export function SiteNav() {
  const { bandId } = useLandingTone();
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    const raf = requestAnimationFrame(onScroll);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('scroll', onScroll);
    };
  }, []);

  const isActive = (l: (typeof LINKS)[number]) => l.match(pathname) || ('band' in l && pathname === '/' && bandId === l.band);

  return (
    <header
      className={`fixed inset-x-0 top-0 z-50 ${scrolled ? 'bg-(--bg)/75 backdrop-blur-xl' : 'bg-transparent'}`}
    >
      <div className="mx-auto flex h-[72px] max-w-[1320px] items-center justify-between gap-6 px-5 text-(--fg) sm:px-8">
        <Link href="/" className="flex items-center gap-2.5 rounded-md">
          <Mark className="h-7 w-7" />
          <span className="text-[16px] font-semibold tracking-[-0.01em]">compresskit</span>
        </Link>

        <nav aria-label="Pages" className="hidden items-center gap-8 text-[14px] md:flex">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              aria-current={isActive(l) ? 'true' : undefined}
              className="relative flex items-center gap-2 rounded-md py-1 opacity-80 transition-opacity hover:opacity-100 aria-[current]:opacity-100"
            >
              <span
                aria-hidden="true"
                className={`h-1.5 w-1.5 rounded-full bg-(--snake) transition-all duration-300 ${
                  isActive(l) ? 'scale-100 opacity-100' : 'scale-0 opacity-0'
                }`}
              />
              {l.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Link
            href="https://github.com/Bhagyesh0603/Model-Compression-ToolKit."
            aria-label="Source on GitHub"
            title="Source on GitHub"
            className="flex h-10 w-10 items-center justify-center rounded-full bg-(--chip) transition-colors hover:bg-(--chip-hover)"
          >
            <GitBranch className="h-4 w-4" aria-hidden="true" />
          </Link>
          <Link
            href="/run"
            className="group inline-flex h-10 items-center gap-1.5 rounded-full bg-(--inv-bg) pl-4 pr-3 text-[14px] font-medium text-(--inv-fg) transition-colors hover:bg-(--inv-hover) sm:pl-5 sm:pr-4"
          >
            <span className="hidden sm:inline">Run it yourself</span>
            <span className="sm:hidden">Run</span>
            <ArrowUpRight
              className="h-4 w-4 transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
              aria-hidden="true"
            />
          </Link>
        </div>
      </div>
    </header>
  );
}
