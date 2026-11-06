# FaceFusion (fork de Christophe)

Fork de FaceFusion 3.9.x (`origin` = github.com/cmxconsulting/facefusion), branche de travail `feature/cmx`.
Une seule branche pour le Mac et le PC : seuls les profils de config diffèrent. Après une mise à jour
de FaceFusion, fusionner `master` dans `feature/cmx`.

## Machines et profils

| Machine | Profil | Lanceurs | Accélération | Encodeur vidéo |
|---|---|---|---|---|
| Mac M2 | `facefusion-mac.ini` | `run.sh`, `run2.sh` (headless), `run3.sh` | `coreml` | `hevc_videotoolbox` |
| PC i9 + RTX 2080 Ti (11 Go), 32 Go RAM, Windows | `facefusion-pc.ini` | `install-pc.bat`, `run-pc.bat`, `headless-pc.bat` | `cuda` | `hevc_nvenc` |

- Le PC est la machine d'encodage : nettement plus rapide (CUDA). La qualité dépend des modèles et réglages, pas de la machine.
- `facefusion.ini` reste identique à `master` (pas de conflit lors des mises à jour). Les réglages sont dans
  les deux profils, chargés avec `--config-path`. Un changement commun (modèle, qualité) se fait **dans les deux profils**.
- Réglages communs : `face_swapper_model = hyperswap_1a_256`, `face_swapper_pixel_boost = 256x256`,
  `output_video_quality = 80`, `output_image_quality = 95`.
- Installation sur le PC : pilote NVIDIA à jour, Python 3.12 et ffmpeg dans le PATH, puis `install-pc.bat`
  (`python install.py --onnxruntime cuda@12 --skip-conda` dans `.venv`). `cuda@13` existe aussi (pilote ≥ 580).
- Sur le Mac, le `.venv` est en Python 3.13 et l'installation passe par `install.sh`.
- Mesurer la vitesse : `python facefusion.py benchmark --execution-providers cuda` (ou `coreml` sur le Mac).
  `benchmark` n'accepte pas `--config-path` et lit `facefusion.ini` : passer les options en ligne de commande.
- Les fichiers `.bat` sont en fins de ligne CRLF : les garder ainsi.

## Modifications maison du code (à conserver lors des fusions avec master)

- `facefusion/content_analyser.py` : analyse de contenu désactivée (`analyse_video` et `detect_nsfw` renvoient `False`).
- `facefusion/core.py` : `common_pre_check()` renvoie `True` (sinon le contrôle de hash du content_analyser bloque le démarrage).

## Vidéos VR180 (chantier en pause)

Contexte complet, état et TODO : `VR180.md` (ne le lire que pour ce sujet).
Code en place mais désactivé par défaut (`--vr180-projection none`) : `facefusion/vr180.py` + accroche dans
`facefusion/workflows/core.py`. Visage proche : monter `face_swapper_pixel_boost` (512x512) aide aussi.

## Règles

- Christophe commit et pousse : ne commit que sur demande explicite, jamais de `git push`.
- Fichiers de test (sources, vidéos, sorties) hors du dépôt : `~/Downloads/tmp/` sur le Mac,
  `%USERPROFILE%\Downloads\tmp\` sur le PC.
- Réponses en français, conclusion d'abord.
