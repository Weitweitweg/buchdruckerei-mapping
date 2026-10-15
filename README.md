# Buchdruckerei – Projection Mapping

Eine 60-Sekunden-Mapping-Show für die Fassade der Buchdruckerei, komplett prozedural in Python erzeugt (OpenCV + NumPy + ffmpeg).

![Szenen](docs/contact_sheet.jpg)

## Inhalt

| Ordner | Was |
|---|---|
| `media/preview_1080p.mp4` | Vorschau 1920×1080 mit Sound – so sieht es auf dem Gebäude aus |
| `media/content_resolume_942x1000.mp4` | Reiner Projektor-Content, deckungsgleich mit `src/facade.jpg` – für Resolume Advanced Output |
| `keyframes/` | Schlüsselbilder der Szenen (Start-/Endbilder für KI-Clips, z. B. Higgsfield / Kling) |
| `docs/masks.jpg` | Masken: Putz, Sandstein, Dach, Glas, Fensterrahmen + Linienzeichnung |
| `src/render.py` | Masken, Szenen und Renderer |
| `src/audio.py` | Soundtrack (100 BPM, synthetisch) |

## Ablauf

| Zeit | Szene |
|---|---|
| 0–6 s | Linienzeichnung baut die Fassade auf, warme Fenster |
| 6–13 s | Neon-Spitzbögen, flackern von der Mitte nach außen an |
| 13–20 s | Pinselwelle: Ocker/Terrakotta, Fenster klappen bunt um, Lichtstreif |
| 20–27 s | Korallenfassade mit wandernden Lichtfalten und Schlagschatten |
| 27–35 s | Goldene Bäume wachsen, bunte Kugeln, Funkeln |
| 35–41 s | Rotierende Ornamente in jedem Fenster |
| 41–47 s | Gebäudeteile blitzen auf, rosa Wireframe |
| 47–50 s | Glitch |
| 50–60 s | Neon-Finale mit Spiralen, Schlusslicht, Blackout |

## Selbst rendern

```bash
pip install opencv-python numpy
cd src
python audio.py                      # -> sound.wav
python render.py stills 9,18,33,55   # Testbilder
python render.py video 0 1800 all    # -> prev_all.mp4 + cont_all.mp4
ffmpeg -i prev_all.mp4 -i sound.wav -c:v copy -c:a aac -shortest preview.mp4
```

## In Resolume

1. `content_resolume_942x1000.mp4` als Clip laden.
2. Im Advanced Output `facade.jpg` als Hintergrund/Referenz nehmen und den Slice per Eckpunkten (oder Mesh-Warp) auf das reale Gebäude ziehen.
3. Da Content und Foto deckungsgleich sind, sitzen Fenster, Bögen und Rahmen automatisch richtig.
