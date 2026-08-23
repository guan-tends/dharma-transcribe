# Test Fixtures

Place short audio/video files here for integration and E2E tests.

## Generating Test Audio

```bash
# 5-second sine wave (for pipeline smoke tests)
ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -ar 16000 -ac 1 tone_5s.wav

# Short speech sample (requires a recording — use your own or royalty-free)
# Keep fixtures under 30 seconds to keep test suite fast.
```

## Naming Convention

- `{language}_{description}.wav` — e.g. `english_teaching_10s.wav`, `tibetan_mantra_5s.wav`
