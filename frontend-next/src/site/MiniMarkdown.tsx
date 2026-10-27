import { Code } from './Code';

/** Just enough markdown for the notebooks' text cells: headings, paragraphs, fenced code, bullet lists. */
export function MiniMarkdown({ source }: { source: string }) {
  const blocks: React.ReactNode[] = [];
  const lines = source.split('\n');
  let i = 0;
  let key = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith('```')) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith('```')) body.push(lines[i++]);
      i++;
      blocks.push(<Code key={key++}>{body.join('\n')}</Code>);
    } else if (/^#{1,6}\s/.test(line)) {
      blocks.push(
        <p key={key++} className="text-[17px] font-medium tracking-[-0.02em]">
          {line.replace(/^#+\s*/, '')}
        </p>,
      );
      i++;
    } else if (/^\s*[-*]\s/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s/.test(lines[i])) items.push(lines[i++].replace(/^\s*[-*]\s/, ''));
      blocks.push(
        <ul key={key++} className="list-disc space-y-1 pl-5">
          {items.map((t) => (
            <li key={t}>{t.replace(/[*`]/g, '')}</li>
          ))}
        </ul>,
      );
    } else if (line.trim() === '') {
      i++;
    } else {
      const para: string[] = [];
      while (i < lines.length && lines[i].trim() !== '' && !lines[i].startsWith('```') && !/^#{1,6}\s/.test(lines[i])) para.push(lines[i++]);
      blocks.push(<p key={key++}>{para.join(' ').replace(/[*`]/g, '')}</p>);
    }
  }
  return <div className="space-y-3 text-[15px] leading-[1.65] text-(--muted)">{blocks}</div>;
}
