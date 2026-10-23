# Third-party components

The archive contains original experiment code and validation outputs. External packages
are installed by pip; their upstream licenses apply. No third-party pretrained neural
weights, scanned environments, human speech recordings or real infant datasets are bundled.

- PyBullet 3.2.7: rigid-body physics and TinyRenderer; https://github.com/bulletphysics/bullet3
- eSpeak NG via espeakng-loader 0.2.4: rule-based formant synthesis and language/voice data;
  https://github.com/espeak-ng/espeak-ng and https://github.com/thewh1teagle/espeakng-loader
  (eSpeak NG is GPL; inspect upstream terms when distributing a combined application).
- PyTorch, NumPy, SciPy, Pillow: learning, arrays, signals and image processing.
- Matplotlib: report plots.
- ImageIO / imageio-ffmpeg: validation video encoding.
- huggingface_hub 0.36.0: authenticated Job submission and private result persistence.

The generated speech waveform is synthetic; its linguistic rules are supplied by the
caregiver environment. This is not equivalent to claiming the synthesizer itself was
invented without human linguistic knowledge.
