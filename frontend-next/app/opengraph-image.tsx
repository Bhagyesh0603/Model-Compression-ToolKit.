import { ImageResponse } from 'next/og';
import { count, models, pct } from '@/src/data/results';

export const alt = 'compresskit: measured CNN compression';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

const m = models.find((x) => x.arch === 'resnet20')!;
const layers = Object.values(m.prune_stats!.channels);

/** Share card drawn from the recorded ResNet-20 run: the same channel columns as the site. */
export default function Image() {
  const max = Math.max(...layers.map(([t]) => t));
  return new ImageResponse(
    (
      <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', background: '#ffffff', color: '#141414', padding: 72 }}>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ fontSize: 26, color: '#6b6b6b', letterSpacing: 2 }}>COMPRESSKIT</div>
          <div style={{ fontSize: 76, lineHeight: 1.02, marginTop: 20, letterSpacing: -3, maxWidth: 900 }}>Shrink a CNN, then measure what it cost.</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6, height: 150 }}>
            {layers.map(([total, kept], i) => (
              <div key={i} style={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', width: 14, height: (total / max) * 150, background: '#e6e6e6' }}>
                <div style={{ width: 14, height: (kept / max) * 150, background: '#1d7a5c' }} />
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', fontSize: 28, color: '#5c5c5c' }}>
            <div>{`ResNet-20: ${count(m.prune_stats!.params_before)} to ${count(m.prune_stats!.params_after)} parameters`}</div>
            <div>{`${pct(m.baseline.accuracy)} to ${pct(m.stages[2].accuracy)} accuracy`}</div>
          </div>
        </div>
      </div>
    ),
    size,
  );
}
