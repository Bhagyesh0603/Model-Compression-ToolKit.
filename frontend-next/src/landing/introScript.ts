export const INTRO_SEEN_KEY = 'compresskit.intro';

/**
 * Inline, render-blocking check for <head>: marks <html data-intro="skip"> when
 * the landing intro has already played this session or motion is reduced, so
 * the curtain never flashes.
 */
export const INTRO_SKIP_SCRIPT = `try{if(sessionStorage.getItem('${INTRO_SEEN_KEY}')||matchMedia('(prefers-reduced-motion: reduce)').matches)document.documentElement.dataset.intro='skip'}catch(e){}`;
