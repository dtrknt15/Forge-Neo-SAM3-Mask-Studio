## Translation PR (`localizations/<locale>.json`)

- **Language / Locale code** (e.g., `zh_CN`, `ko_KR`, `es_ES`):
- **File added/updated**: `localizations/<locale>.json`

### Checklist
- [ ] File is valid UTF-8 JSON with no trailing commas.
- [ ] All JSON keys match `localizations/ja_JP.json` (only values are translated).
- [ ] Format placeholders such as `{width}`, `{height}`, `{count}`, `{px}`, `{path}`, `{error}` are preserved intact.
- [ ] `preset.*` keys translate display names only (`presets.json` prompts remain in English).
