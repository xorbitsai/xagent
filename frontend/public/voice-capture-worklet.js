/* AudioWorklet: keep microphone capture off the UI thread. VAD runs separately. */
class VoiceCapture extends AudioWorkletProcessor {
  constructor() { super(); this.block = new Float32Array(1024); this.offset = 0; }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    for (const sample of input) {
      this.block[this.offset++] = sample;
      if (this.offset === this.block.length) {
        this.port.postMessage(this.block, [this.block.buffer]);
        this.block = new Float32Array(1024); this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor("xagent-voice-capture", VoiceCapture);
