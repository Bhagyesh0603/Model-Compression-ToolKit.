/**
 * A TypeScript port of compresskit's per-layer SIMD alignment
 * (`compresskit/pruning/prune.py::_aligned_ratio`). The unit test replays 180
 * (width, ratio, align) cases recorded from the Python function and requires an
 * exact match, so the explorer on the pruning page cannot drift from the code.
 */

/** Python's round(): halves go to the even neighbour (Math.round rounds them up). */
export function pyRound(x: number): number {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

export interface AlignedKeep {
  kept: number;
  /** The alignment actually used: shrinks for layers too narrow to fit two aligned blocks. */
  alignUsed: number;
  effectiveRatio: number;
}

export function alignedKeep(width: number, ratio: number, align: number): AlignedKeep {
  if (width <= 0 || ratio <= 0) return { kept: width, alignUsed: align, effectiveRatio: 0 };

  let alignUsed = align && align > 1 ? align : 1;
  while (alignUsed > 1 && Math.floor(width / alignUsed) < 2) alignUsed = Math.floor(alignUsed / 2);

  if (alignUsed <= 1) {
    const kept = Math.max(1, Math.min(width, pyRound(width * (1 - ratio))));
    return { kept, alignUsed, effectiveRatio: 1 - kept / width };
  }

  const desiredKept = width * (1 - ratio);
  const maxBlocks = Math.floor(width / alignUsed);
  const blocks = Math.min(maxBlocks, Math.max(1, pyRound(desiredKept / alignUsed)));
  const kept = blocks * alignUsed;
  return { kept, alignUsed, effectiveRatio: 1 - kept / width };
}
