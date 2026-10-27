type Props = {
  className?: string;
  title?: string;
};

/**
 * compresskit mark: four channels, two kept (solid) and two pruned (outline
 * only, still drawn so you can see what was removed). Inherits colour from
 * `currentColor`.
 */
export function Mark({ className, title }: Props) {
  const bars = [
    { x: 12, kept: true },
    { x: 39, kept: false },
    { x: 66, kept: true },
    { x: 93, kept: false },
  ];
  return (
    <svg
      viewBox="0 0 120 120"
      className={className}
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : true}
      aria-label={title}
    >
      {bars.map(({ x, kept }) => (
        <rect
          key={x}
          x={kept ? x : x + 2.5}
          y={kept ? 8 : 10.5}
          width={kept ? 15 : 10}
          height={kept ? 104 : 99}
          rx={kept ? 7.5 : 5}
          fill={kept ? 'currentColor' : 'none'}
          stroke={kept ? undefined : 'currentColor'}
          strokeWidth={kept ? undefined : 5}
          strokeDasharray={kept ? undefined : '6 6'}
        />
      ))}
    </svg>
  );
}
