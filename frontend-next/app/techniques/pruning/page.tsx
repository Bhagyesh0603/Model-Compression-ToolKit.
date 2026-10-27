import type { Metadata } from 'next';
import groups from '@/src/data/groups.json';
import { modelBySlug, pct, count } from '@/src/data/results';
import { StripReplay } from '@/src/explore/StripReplay';
import { ChannelBars } from '@/src/site/charts/ChannelBars';
import { AlignExplorer } from '@/src/site/charts/AlignExplorer';
import { Provenance } from '@/src/site/Provenance';
import { Figure, Section } from '@/src/site/Section';
import { TechniqueShell } from '@/src/site/TechniqueShell';

export const metadata: Metadata = { title: 'Structured pruning — compresskit' };

export default function PruningPage() {
  const m = modelBySlug('resnet20');
  if (!m?.prune_stats) return null;
  const rows = Object.entries(m.prune_stats.channels)
    .map(([name, [before, after]]) => ({ name, before, after }));
  const stages = [
    { title: 'Stage 1', rows: rows.filter((r) => r.name === 'conv1' || r.name.startsWith('layer1.')) },
    { title: 'Stage 2', rows: rows.filter((r) => r.name.startsWith('layer2.')) },
    { title: 'Stage 3', rows: rows.filter((r) => r.name.startsWith('layer3.')) },
  ];
  const ps = m.prune_stats;
  // A group of more than two layers is a residual connection; two is just a conv with its BatchNorm.
  const residual = groups.groups.filter((g) => g.length > 2);
  const pairs = groups.groups.length - residual.length;

  return (
    <TechniqueShell
      slug="pruning"
      title="Structured pruning removes whole channels"
      lede={
        <p>
          Zeroing single weights leaves every tensor the same size, and no maintained CPU runtime turns that sparsity
          into speed. Removing a channel shrinks the actual matrix multiply, which is where CPU latency comes from.
        </p>
      }
    >
      <Section
        title="What was removed"
        lede={
          <p>
            One recorded run: {m.name} on {m.dataset}, asking for {pct(ps.ratio_requested, 0)} of channels in {ps.steps}{' '}
            steps. Parameters went from {count(ps.params_before)} to {count(ps.params_after)}. A channel is a whole
            filter, so removing half the channels in a layer removes about three quarters of that layer&apos;s weights.
          </p>
        }
      >
        <div className="mb-6">
          <Figure caption="Each column is a layer and each cell a channel. The recorded 50% cut plays when this scrolls into view.">
            <StripReplay />
          </Figure>
        </div>
        <Figure
          caption={
            <>
              Pale bar: original width. Solid: kept. An asterisk marks a kept count that is not a multiple of 16, where
              the layer was too narrow to hold two aligned blocks and its alignment shrank. The classifier (fc) is never
              pruned.
            </>
          }
        >
          <div className="grid gap-x-10 gap-y-8 lg:grid-cols-3">
            {stages.map((st) => (
              <div key={st.title}>
                <p className="mb-3 text-[12px] font-medium uppercase tracking-[0.06em] text-(--muted)">{st.title}</p>
                <ChannelBars rows={st.rows} align={16} />
              </div>
            ))}
          </div>
          <Provenance p={m.provenance} source={m.source} />
        </Figure>
      </Section>

      <Section tone="dark"
        title="Layers that must agree"
        lede={
          <p>
            A residual connection adds two tensors, so every layer feeding the same addition has to keep the same
            channel positions. Torch-Pruning&apos;s dependency graph finds these groups. They share a single ratio, and if
            two members were given different ratios the gentler one wins.
          </p>
        }
      >
        <Figure
          caption={`${m.name} has ${residual.length} residual groups of ${residual.map((g) => g.length).join(', ')} layers. The other ${pairs} groups are a single conv with its BatchNorm and constrain nothing else.`}
        >
          <ul className="grid gap-3 md:grid-cols-3">
            {residual.map((g, i) => (
              <li key={i} className="rounded-2xl bg-(--chip) p-4">
                <p className="text-[12px] text-(--muted)">Residual group {i + 1}: {g.length} layers share one ratio</p>
                <p className="mt-2 flex flex-wrap gap-1.5 font-mono text-[11px]">
                  {g.map((layer) => (
                    <span key={layer} className="rounded-md border border-(--line) px-1.5 py-0.5">
                      {layer}
                    </span>
                  ))}
                </p>
              </li>
            ))}
          </ul>
          <Provenance p={groups.provenance as never} />
        </Figure>
      </Section>

      <Section
        title="Why widths snap to a multiple"
        lede={
          <p>
            Vector units process channels in blocks of 8, 16 or 32, so a width that fills whole blocks wastes no lanes.
            Each layer rounds its kept count to the nearest achievable multiple of the requested alignment. A layer too
            narrow to hold two aligned blocks halves its own alignment instead of relaxing it for the whole network, and
            a layer is never reduced to zero channels.
          </p>
        }
      >
        <Figure
          caption="The same function as compresskit.pruning.prune._aligned_ratio, ported to TypeScript and checked against 180 recorded Python results by a unit test. The rule is deterministic, so this explorer is exact, not a simulation."
        >
          <AlignExplorer />
        </Figure>
      </Section>
    </TechniqueShell>
  );
}
