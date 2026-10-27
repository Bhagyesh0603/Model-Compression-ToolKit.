/** The five honesty rules, as stated in README.md ("The honesty rules"). */
export const HONESTY_RULES = [
  {
    title: 'Latency is measured',
    body: 'FP32, FP16 and INT8 are timed through an identical ONNX Runtime session: same threads, batch, warm-up and run counts. Never estimated from FLOPs.',
  },
  {
    title: 'Calibration stays on train',
    body: 'BatchNorm re-estimation and quantization calibration both fit parameters to data, so they only ever see the training split.',
  },
  {
    title: 'Accuracy of the exported file',
    body: 'Top-1 is measured on the .onnx artifact, so export bugs and quantization error show up in the number instead of hiding.',
  },
  {
    title: 'Float ops are reported',
    body: 'An op with no INT8 kernel stays float32. The file still shrinks, so the report counts those ops rather than letting a size win pose as a speed win.',
  },
  {
    title: 'FP16 is a size win',
    body: 'On the CPU provider FP16 is cast back to float32 to compute. It is labelled a size win, not a speed win, because that is what it measures as.',
  },
] as const;
