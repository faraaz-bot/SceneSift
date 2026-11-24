"""End-to-end orchestration: analyse a video once, then score, select, and render per request."""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import export, segmentation
from .config import METHOD_DIVERSITY, METHOD_FEATURES, Config
from .features import audio as audio_features
from .features import classical as classical_features
from .features import motion as motion_features
from .features import speech as speech_features
from .features import visual as visual_features
from .frames import can_decode
from .frames import iter_frames as _frames
from .media import format_duration, inspect_video
from .scoring import combine, fuse, robust_unit
from .selection import select
from .datatypes import Clip, Segment, Utterance

Progress = Callable[[float, str], None]
_MODELS: dict[tuple, object] = {}
CACHE_VERSION = 5  # bump when the stored analysis changes shape

EVIDENCE_LABELS = {
    "visual": "visual content",
    "motion": "motion",
    "audio": "audio activity",
    "speech": "speech content",
    "novelty": "new content",
    "colour": "distinct colours",
    "residual": "movement in the scene",
    "moving": "area in motion",
    "camera": "camera movement",
    "edges": "edge detail",
}


@dataclass
class Analysis:
    """Prompt-independent measurements of one video."""

    path: str
    info: dict
    cuts: list[float]
    visual_times: np.ndarray
    visual_embeddings: np.ndarray
    visual_encoder: str
    motion_times: np.ndarray
    motion: np.ndarray
    detail: np.ndarray  # grey-level spread of each motion sample; near zero for blank frames
    # Classical cues per motion sample (see features/classical.py); movement cues have cuts masked.
    cues: dict[str, np.ndarray]
    colour_embeddings: np.ndarray  # colour-histogram descriptor of each visual sample
    audio: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    utterances: list[Utterance] | None  # None: speech not analysed; []: analysed, none found
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(self.info["duration_seconds"])


@dataclass
class Scored:
    segments: list[Segment]
    features: dict[str, np.ndarray | None]
    score: np.ndarray
    contributions: dict[str, np.ndarray]
    weights: dict[str, float]
    embeddings: np.ndarray
    transcripts: list[str]
    notes: list[str]


@dataclass
class Summary:
    video: str
    method: str
    prompt: str | None
    target_seconds: float
    clips: list[dict]
    total_seconds: float
    source_seconds: float
    redundancy: float | None
    weights: dict[str, float]
    evidence_missing: list[str]
    timings: dict[str, float]
    notes: list[str]
    output_path: str | None = None
    json_path: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def intervals(self) -> list[tuple[float, float]]:
        return [(clip["start"], clip["end"]) for clip in self.clips]

    def describe(self) -> str:
        used = ", ".join(f"{EVIDENCE_LABELS[name]} {weight:.0%}" for name, weight in self.weights.items())
        lines = [
            f"Selected {len(self.clips)} clip(s) totalling {format_duration(self.total_seconds)} "
            f"from a {format_duration(self.source_seconds)} source ({self.method} method).",
            f"Evidence used: {used}.",
        ]
        if self.evidence_missing:
            lines.append("Not available for this video: "
                         + ", ".join(EVIDENCE_LABELS[name] for name in self.evidence_missing) + ".")
        return " ".join(lines)


def _report(progress: Progress | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(min(max(fraction, 0.0), 1.0), message)


def _model(key: tuple, build: Callable[[], object]):
    if key not in _MODELS:
        _MODELS[key] = build()
    return _MODELS[key]


def _visual_encoder(cfg: Config, warnings: list[str]):
    key = ("visual", cfg.visual_backend, cfg.clip_model, cfg.clip_pretrained, cfg.device)
    return _model(key, lambda: visual_features.load_visual_encoder(cfg, warnings))


def _text_encoder(cfg: Config, warnings: list[str]):
    if cfg.text_backend == "none":
        return None
    key = ("text", cfg.text_model, cfg.device)
    try:
        return _model(key, lambda: speech_features.TextEncoder(cfg.text_model, visual_features.resolve_device(cfg)))
    except (ImportError, OSError, RuntimeError) as exc:
        if cfg.text_backend != "auto":
            raise
        warnings.append(f"Sentence embeddings unavailable ({type(exc).__name__}: {exc}); "
                        "transcript content is not scored.")
        return None


def _cache_file(path: str, cfg: Config) -> Path:
    stat = Path(path).stat()
    identity = f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    params = json.dumps([CACHE_VERSION, cfg.motion_fps, cfg.visual_fps, cfg.analysis_width, cfg.shot_backend,
                         cfg.visual_backend, cfg.clip_model, cfg.clip_pretrained, cfg.speech_backend,
                         cfg.whisper_model, cfg.whisper_vad])
    digest = hashlib.sha1(f"{identity}|{params}".encode()).hexdigest()[:20]
    return Path(cfg.cache_dir).expanduser() / f"{digest}.npz"


def _save(analysis: Analysis, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "info": analysis.info, "cuts": analysis.cuts, "visual_encoder": analysis.visual_encoder,
        "utterances": None if analysis.utterances is None
        else [[u.start, u.end, u.text] for u in analysis.utterances],
        "timings": analysis.timings, "warnings": analysis.warnings, "has_audio": analysis.audio is not None,
    }
    empty = np.zeros(0)
    times, energy, flux = analysis.audio if analysis.audio is not None else (empty, empty, empty)
    np.savez_compressed(file, meta=json.dumps(meta), visual_times=analysis.visual_times,
                        visual_embeddings=analysis.visual_embeddings, motion_times=analysis.motion_times,
                        motion=analysis.motion, detail=analysis.detail,
                        **{f"cue_{name}": values for name, values in analysis.cues.items()},
                        colour_embeddings=analysis.colour_embeddings, audio_times=times,
                        audio_energy=energy, audio_flux=flux)


def _load(path: str, file: Path) -> Analysis:
    with np.load(file, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        audio = (data["audio_times"], data["audio_energy"], data["audio_flux"]) if meta["has_audio"] else None
        utterances = None if meta["utterances"] is None else [Utterance(*u) for u in meta["utterances"]]
        return Analysis(path=path, info=meta["info"], cuts=meta["cuts"], visual_times=data["visual_times"],
                        visual_embeddings=data["visual_embeddings"], visual_encoder=meta["visual_encoder"],
                        motion_times=data["motion_times"], motion=data["motion"], detail=data["detail"],
                        cues={key[4:]: data[key] for key in data.files if key.startswith("cue_")},
                        colour_embeddings=data["colour_embeddings"],
                        audio=audio, utterances=utterances, timings=meta["timings"],
                        warnings=meta["warnings"])


def analyze(video_path: str, cfg: Config | None = None, progress: Progress | None = None) -> Analysis:
    """Decode the video once and measure every modality. Results are cached per file and settings."""
    cfg = cfg or Config()
    video_path = str(video_path)
    info = inspect_video(video_path)
    cache = _cache_file(video_path, cfg)
    if cfg.use_cache and cache.is_file():
        _report(progress, 0.9, "Loaded cached analysis")
        return _load(video_path, cache)

    warnings: list[str] = []
    timings: dict[str, float] = {}
    duration = float(info["duration_seconds"])
    degraded = False

    # OpenCV's bundled decoders do not cover every codec (Theora, for one). When it cannot read
    # the file, analyse a temporary H.264 copy made by FFmpeg; the timeline is unchanged.
    frames_path = video_path
    proxy_dir = None
    if not can_decode(video_path):
        proxy_dir = tempfile.TemporaryDirectory(prefix="scenesift-proxy-")
        _report(progress, 0.0, "Converting video for analysis")
        try:
            frames_path = str(export.make_proxy(video_path, Path(proxy_dir.name) / "proxy.mp4"))
        except RuntimeError:
            proxy_dir.cleanup()
            raise
        warnings.append("This video's codec could not be read directly, so a temporary converted copy "
                        "was analysed.")
    try:
        # One decoding pass feeds motion, the fallback cut detector, and the visual encoder.
        encoder = _visual_encoder(cfg, warnings)
        degraded |= cfg.visual_backend == "auto" and encoder.name == "histogram"
        colour_encoder = visual_features.HistogramEncoder()
        motion_times, motion, differences, detail = [], [], [], []
        cues: dict[str, list[float]] = {"edges": [], "residual": [], "moving": [], "camera": []}
        aligner = classical_features.FrameAligner()
        visual_times, embeddings, colour, batch = [], [], [], []
        previous = None
        next_visual = 0.0
        spent = {"motion": 0.0, "visual": 0.0, "classical": 0.0}
        pass_started = time.perf_counter()
        for moment, frame in _frames(frames_path, cfg.motion_fps):
            tick = time.perf_counter()
            gray = motion_features.to_analysis_gray(frame, cfg.analysis_width)
            motion_times.append(moment)
            detail.append(float(gray.std()))
            if previous is None:
                motion.append(np.nan)
                differences.append(np.nan)
            else:
                motion.append(motion_features.local_motion(previous, gray))
                differences.append(motion_features.histogram_difference(previous, gray))
            spent["motion"] += time.perf_counter() - tick
            tick = time.perf_counter()
            cues["edges"].append(classical_features.edge_density(gray))
            movement = aligner.measure(gray)
            for name in ("residual", "moving", "camera"):
                cues[name].append(getattr(movement, name))
            spent["classical"] += time.perf_counter() - tick
            previous = gray
            if moment + 1e-9 >= next_visual:
                tick = time.perf_counter()
                prepared = visual_features.prepare_frame(frame)
                colour.append(colour_encoder.encode_images([prepared])[0])
                spent["classical"] += time.perf_counter() - tick
                tick = time.perf_counter()
                visual_times.append(moment)
                batch.append(prepared)
                while next_visual <= moment + 1e-9:
                    next_visual += 1.0 / cfg.visual_fps
                if len(batch) == 32:
                    embeddings.append(encoder.encode_images(batch))
                    batch = []
                spent["visual"] += time.perf_counter() - tick
            _report(progress, 0.6 * moment / max(duration, 1e-6), "Reading frames: motion and visual features")
        if batch:
            tick = time.perf_counter()
            embeddings.append(encoder.encode_images(batch))
            spent["visual"] += time.perf_counter() - tick
        if not visual_times:
            raise ValueError("No frames could be decoded from the video.")
        timings["decode"] = time.perf_counter() - pass_started - sum(spent.values())
        timings.update(spent)
        motion_times_arr = np.asarray(motion_times)
        motion_arr = np.asarray(motion, dtype=float)

        started = time.perf_counter()
        _report(progress, 0.62, "Detecting shot changes")
        cuts = None
        if cfg.shot_backend in ("auto", "scenedetect"):
            try:
                cuts = segmentation.detect_cuts_scenedetect(frames_path)
            except ImportError:
                if cfg.shot_backend == "scenedetect":
                    raise
        if cuts is None:
            cuts = segmentation.cuts_from_differences(motion_times_arr, np.asarray(differences, dtype=float))
        timings["shots"] = time.perf_counter() - started

        started = time.perf_counter()
        audio = None
        samples = None
        if info["has_audio"]:
            _report(progress, 0.66, "Measuring audio")
            samples = audio_features.load_audio(video_path)
            if samples is not None:
                audio = audio_features.audio_signals(samples)
        if audio is None:
            warnings.append("No usable audio: audio and speech evidence are skipped."
                            if not info["has_audio"] or samples is None
                            else "The audio track is silent: audio and speech evidence are skipped.")
        timings["audio"] = time.perf_counter() - started

        utterances = None
        timings["speech"] = 0.0
        if audio is not None and cfg.speech_backend != "none":
            _report(progress, 0.7, "Transcribing speech")
            try:
                device = visual_features.resolve_device(cfg)
                model = _model(("whisper", cfg.whisper_model, device),
                               lambda: speech_features.load_transcriber(cfg, device))
                started = time.perf_counter()  # model loading is excluded, as it is for the visual encoder
                utterances = speech_features.transcribe(model, samples, cfg.whisper_vad)
                timings["speech"] = time.perf_counter() - started
            except (ImportError, OSError, RuntimeError) as exc:
                if cfg.speech_backend != "auto":
                    raise
                degraded = True
                warnings.append(f"Speech transcription unavailable ({type(exc).__name__}: {exc}).")
            if utterances == []:
                warnings.append("No speech was detected: transcript evidence is skipped.")

        analysis = Analysis(path=video_path, info=info, cuts=[float(c) for c in cuts],
                            visual_times=np.asarray(visual_times), visual_embeddings=np.vstack(embeddings),
                            visual_encoder=encoder.name, motion_times=motion_times_arr,
                            motion=motion_features.mask_cuts(motion_times_arr, motion_arr, cuts),
                            detail=np.asarray(detail, dtype=float),
                            cues={name: np.asarray(values, dtype=float) if name == "edges"
                                  else motion_features.mask_cuts(motion_times_arr, np.asarray(values, dtype=float),
                                                                 cuts)
                                  for name, values in cues.items()},
                            colour_embeddings=np.vstack(colour), audio=audio,
                            utterances=utterances, timings=timings, warnings=warnings)
        if cfg.use_cache and not degraded:
            _save(analysis, cache)
        _report(progress, 0.9, "Analysis complete")
        return analysis
    finally:
        if proxy_dir is not None:
            proxy_dir.cleanup()




def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _within(times: np.ndarray, start: float, end: float) -> np.ndarray:
    return np.flatnonzero((times >= start) & (times < end))


def score(analysis: Analysis, cfg: Config | None = None, prompt: str | None = None,
          method: str = "multimodal", drop: tuple[str, ...] = ()) -> Scored:
    """Build candidates, compute per-candidate evidence, and fuse it for the chosen method."""
    cfg = cfg or Config()
    if method not in METHOD_FEATURES:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHOD_FEATURES)}.")
    prompt = (prompt or "").strip() or None
    notes: list[str] = []
    segments = segmentation.build_candidates(analysis.duration, analysis.cuts, cfg)
    segments = segmentation.drop_blank(segments, analysis.motion_times, analysis.detail,
                                       cfg.blank_frame_std, cfg.max_blank_fraction)
    count = len(segments)
    frames = analysis.visual_embeddings
    times = analysis.visual_times
    video_mean = _unit(frames.mean(axis=0))

    prompt_similarity = None
    if prompt and analysis.visual_encoder != "histogram":
        encoder = _visual_encoder(cfg, notes)
        if encoder.supports_text and encoder.name == analysis.visual_encoder:
            prompt_similarity = frames @ encoder.encode_text([prompt])[0]
    if prompt and prompt_similarity is None:
        notes.append("The request could not guide visual scoring (no language-grounded visual model).")

    embeddings = np.zeros((count, frames.shape[1]), dtype=np.float32)
    distinct = np.full(count, np.nan)
    relevance = np.full(count, np.nan)
    visual_novelty = np.full(count, np.nan)
    motion = np.full(count, np.nan)
    colours = analysis.colour_embeddings
    colour_mean = _unit(colours.mean(axis=0))
    colour_distinct = np.full(count, np.nan)
    colour_novelty = np.full(count, np.nan)
    cues = {name: np.full(count, np.nan) for name in analysis.cues}
    for index, segment in enumerate(segments):
        inside = _within(times, segment.start, segment.end)
        if inside.size == 0:
            inside = np.array([int(np.argmin(np.abs(times - (segment.start + segment.end) / 2.0)))])
        embeddings[index] = _unit(frames[inside].mean(axis=0))
        distinct[index] = 1.0 - float(embeddings[index] @ video_mean)
        if prompt_similarity is not None:
            top = np.sort(prompt_similarity[inside])[-cfg.prompt_top_k:]
            relevance[index] = float(top.mean())
        before = _within(times, segment.start - cfg.context_window_seconds, segment.start)
        if before.size:
            visual_novelty[index] = 1.0 - float(embeddings[index] @ _unit(frames[before].mean(axis=0)))
        colour_vector = _unit(colours[inside].mean(axis=0))
        colour_distinct[index] = 1.0 - float(colour_vector @ colour_mean)
        if before.size:
            colour_novelty[index] = 1.0 - float(colour_vector @ _unit(colours[before].mean(axis=0)))
        dense = _within(analysis.motion_times, segment.start, segment.end)
        for values, target in ((analysis.motion, motion), *((analysis.cues[name], cues[name]) for name in cues)):
            if np.isfinite(values[dense]).any():
                target[index] = float(np.nanmean(values[dense]))

    audio = None
    if analysis.audio is not None:
        audio_times, energy, flux = analysis.audio
        mean_energy = np.full(count, np.nan)
        peak_change = np.full(count, np.nan)
        for index, segment in enumerate(segments):
            inside = _within(audio_times, segment.start, segment.end)
            if inside.size:
                mean_energy[index] = float(energy[inside].mean())
                peak_change[index] = float(np.percentile(flux[inside], 90))
        audio = combine([robust_unit(mean_energy), robust_unit(peak_change)])

    speech = None
    topic = None
    transcripts = [""] * count
    if analysis.utterances:
        utterances = analysis.utterances
        for index, segment in enumerate(segments):
            transcripts[index] = " ".join(u.text for u in utterances
                                          if min(u.end, segment.end) - max(u.start, segment.start) > 0.25)
        text_encoder = _text_encoder(cfg, notes)
        if text_encoder is not None:
            vectors = text_encoder.encode([u.text for u in utterances])
            if prompt:
                importance = vectors @ text_encoder.encode([prompt])[0]
            else:
                importance = vectors @ _unit(vectors.mean(axis=0))  # centrality within the transcript
            novelty = speech_features.topic_novelty(utterances, vectors, cfg.context_window_seconds)
            raw_speech = np.full(count, np.nan)
            raw_topic = np.full(count, np.nan)
            for index, segment in enumerate(segments):
                hits = [j for j, u in enumerate(utterances)
                        if min(u.end, segment.end) - max(u.start, segment.start)
                        >= min(0.5, 0.5 * (u.end - u.start))]
                if hits:
                    raw_speech[index] = float(np.max(importance[hits]))
                    if np.isfinite(novelty[hits]).any():
                        raw_topic[index] = float(np.nanmean(novelty[hits]))
            speech = robust_unit(raw_speech)
            topic = robust_unit(raw_topic)

    features: dict[str, np.ndarray | None] = {
        "visual": robust_unit(relevance if prompt_similarity is not None else distinct),
        "motion": robust_unit(motion) if np.isfinite(motion).any() else None,
        "audio": audio,
        "speech": speech,
        "novelty": combine([robust_unit(visual_novelty), topic]),
        "colour": combine([robust_unit(colour_distinct), robust_unit(colour_novelty)]),
        **{name: robust_unit(values) if np.isfinite(values).any() else None for name, values in cues.items()},
    }
    use = [name for name in METHOD_FEATURES[method] if name not in drop]
    fused, contributions, weights = fuse(features, cfg.weights_for(method), use)
    return Scored(segments=segments, features=features, score=fused, contributions=contributions,
                  weights=weights, embeddings=embeddings, transcripts=transcripts, notes=notes)


def importance_curve(scored: Scored, duration: float, step: float, count: int | None = None) -> np.ndarray:
    """The method's score on a regular time grid, for rank correlation with annotator scores.

    Bin k covers [k * step, (k + 1) * step). Its value is the mean score of the candidates that
    contain the bin's centre; bins no candidate covers get the lowest score in the video.
    """
    count = int(np.ceil(duration / step - 1e-9)) if count is None else count
    centres = (np.arange(count) + 0.5) * step
    total = np.zeros(count)
    hits = np.zeros(count)
    for segment, value in zip(scored.segments, scored.score):
        inside = (centres >= segment.start) & (centres < segment.end)
        total[inside] += value
        hits[inside] += 1
    floor = float(np.min(scored.score)) if len(scored.score) else 0.0
    return np.where(hits > 0, total / np.maximum(hits, 1), floor)


def _clip_record(clip: Clip, scored: Scored) -> dict:
    lead = max(clip.members, key=lambda i: scored.score[i])
    evidence = {name: round(float(values[lead]), 4) for name, values in scored.contributions.items()}
    strengths = {name: float(np.nan_to_num(scored.features[name][lead], nan=0.5)) for name in evidence}
    strong = [EVIDENCE_LABELS[name] for name, value in sorted(strengths.items(), key=lambda kv: -kv[1])
              if value >= 0.6][:3]
    rationale = "Above this video's norm for " + ", ".join(strong) if strong else "Highest combined score available"
    transcript = scored.transcripts[lead]
    return {"start": round(clip.start, 3), "end": round(clip.end, 3), "score": round(clip.score, 4),
            "evidence": evidence, "rationale": rationale,
            "transcript": transcript[:240] + ("…" if len(transcript) > 240 else "")}


def _redundancy(clips: list[Clip], analysis: Analysis) -> float | None:
    """Mean pairwise cosine similarity between the selected clips' visual embeddings."""
    if len(clips) < 2:
        return None
    vectors = []
    for clip in clips:
        inside = _within(analysis.visual_times, clip.start, clip.end)
        if inside.size:
            vectors.append(_unit(analysis.visual_embeddings[inside].mean(axis=0)))
    if len(vectors) < 2:
        return None
    matrix = np.vstack(vectors) @ np.vstack(vectors).T
    upper = matrix[np.triu_indices(len(vectors), k=1)]
    return float(np.clip(upper, 0.0, 1.0).mean())


def summarize(video_path: str, *, prompt: str | None = None, target_seconds: float = 60.0,
              method: str = "multimodal", cfg: Config | None = None, out_dir: str | Path | None = None,
              render: bool = True, drop: tuple[str, ...] = (), diversity: bool | None = None,
              analysis: Analysis | None = None, scored: Scored | None = None,
              progress: Progress | None = None) -> Summary:
    """Produce a highlight selection (and, with `render`, an MP4) for one video.

    Pass `scored` (from `score` with the same arguments) to reuse scores already computed.
    """
    cfg = cfg or Config()
    if analysis is None:
        analysis = analyze(video_path, cfg, progress)
    started = time.perf_counter()
    if scored is None:
        scored = score(analysis, cfg, prompt, method, drop)
    use_diversity = METHOD_DIVERSITY[method] if diversity is None else diversity
    clips = select(scored.segments, scored.score, float(target_seconds), analysis.duration,
                   embeddings=scored.embeddings, diversity=use_diversity, mmr_lambda=cfg.mmr_lambda,
                   max_overlap=cfg.max_overlap, context_before=cfg.context_before,
                   context_after=cfg.context_after)
    timings = dict(analysis.timings)
    timings["score_select"] = time.perf_counter() - started

    notes = list(dict.fromkeys([*analysis.warnings, *scored.notes]))
    if target_seconds >= analysis.duration:
        notes.append("The requested length is at least the source length, so the whole video is returned.")
    wanted = [name for name in METHOD_FEATURES[method]
              if name not in drop and cfg.weights_for(method).get(name, 0.0) > 0.0]
    summary = Summary(
        video=str(video_path), method=method, prompt=(prompt or "").strip() or None,
        target_seconds=float(target_seconds), clips=[_clip_record(clip, scored) for clip in clips],
        total_seconds=round(sum(clip.length for clip in clips), 3), source_seconds=analysis.duration,
        redundancy=_redundancy(clips, analysis), weights={k: round(v, 4) for k, v in scored.weights.items()},
        evidence_missing=[name for name in wanted if name not in scored.weights], timings=timings, notes=notes)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if render:
            _report(progress, 0.93, "Rendering highlight reel")
            started = time.perf_counter()
            summary.output_path = str(export.render(str(video_path), clips, out_dir / "highlights.mp4",
                                                    has_audio=bool(analysis.info["has_audio"]),
                                                    crf=cfg.crf, preset=cfg.preset))
            summary.timings["render"] = time.perf_counter() - started
        json_path = out_dir / "highlights.json"
        summary.json_path = str(json_path)
        json_path.write_text(json.dumps({**summary.to_dict(), "config": cfg.to_dict()}, indent=2),
                             encoding="utf-8")
    _report(progress, 1.0, "Done")
    return summary
