/** A command or code block. Horizontal scroll instead of wrapping so commands stay copyable. */
export function Code({ children, label }: { children: string; label?: string }) {
  return (
    <figure className="overflow-hidden rounded-2xl border border-(--line)">
      {label ? <figcaption className="border-b border-(--line) bg-(--chip) px-4 py-2 text-[12px] text-(--muted)">{label}</figcaption> : null}
      <pre className="overflow-x-auto p-4 font-mono text-[12.5px] leading-[1.65]">
        <code>{children}</code>
      </pre>
    </figure>
  );
}
