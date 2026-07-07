"""Link per-scene face + speaker clusters into unified character identities.

Faces are the primary anchor for a character. Each diarized speaker cluster is
linked to the face it most consistently co-occurs with (strongest signal:
scenes with a single face and single speaker). Speakers that never co-occur
with a face become voice-only characters (e.g. narrators / off-screen voices).
"""

from __future__ import annotations

from collections import defaultdict

from ai_cowatcher.domain import CharacterIdentity, SceneEventRecord


def link_identities(
    title_id: str,
    scenes: list[SceneEventRecord],
    *,
    min_cooccur: int = 1,
) -> list[CharacterIdentity]:
    ordered = sorted(scenes, key=lambda s: s.start_ts)

    face_first_ts: dict[str, float] = {}
    speaker_first_ts: dict[str, float] = {}
    strong_pairs: dict[tuple[str, str], int] = defaultdict(int)
    weak_pairs: dict[tuple[str, str], int] = defaultdict(int)

    for scene in ordered:
        faces = list(dict.fromkeys(scene.face_cluster_ids))
        speakers = list(dict.fromkeys(scene.speaker_cluster_ids))
        for face in faces:
            face_first_ts.setdefault(face, scene.start_ts)
        for speaker in speakers:
            speaker_first_ts.setdefault(speaker, scene.start_ts)
        for face in faces:
            for speaker in speakers:
                weak_pairs[(face, speaker)] += 1
        if len(faces) == 1 and len(speakers) == 1:
            strong_pairs[(faces[0], speakers[0])] += 1

    speaker_to_face = _assign_speakers_to_faces(
        faces=list(face_first_ts),
        speakers=list(speaker_first_ts),
        face_first_ts=face_first_ts,
        strong_pairs=strong_pairs,
        weak_pairs=weak_pairs,
        min_cooccur=min_cooccur,
    )

    face_to_speakers: dict[str, list[str]] = defaultdict(list)
    for speaker, face in speaker_to_face.items():
        face_to_speakers[face].append(speaker)

    provisional: list[tuple[float, list[str], list[str]]] = []
    for face in face_first_ts:
        linked_speakers = sorted(face_to_speakers.get(face, []))
        first_ts = min(
            [face_first_ts[face]] + [speaker_first_ts[s] for s in linked_speakers]
        )
        provisional.append((first_ts, [face], linked_speakers))

    for speaker in speaker_first_ts:
        if speaker in speaker_to_face:
            continue
        provisional.append((speaker_first_ts[speaker], [], [speaker]))

    provisional.sort(key=lambda item: (item[0], item[1], item[2]))

    identities: list[CharacterIdentity] = []
    for index, (first_ts, faces, speakers) in enumerate(provisional):
        identities.append(
            CharacterIdentity(
                character_id=f"{title_id}-char-{index:03d}",
                title_id=title_id,
                name=None,
                face_cluster_ids=tuple(faces),
                speaker_cluster_ids=tuple(speakers),
                first_ts=first_ts,
            )
        )
    return identities


def _assign_speakers_to_faces(
    *,
    faces: list[str],
    speakers: list[str],
    face_first_ts: dict[str, float],
    strong_pairs: dict[tuple[str, str], int],
    weak_pairs: dict[tuple[str, str], int],
    min_cooccur: int,
) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for speaker in speakers:
        best_face: str | None = None
        best_score: tuple[int, float] | None = None
        for face in faces:
            score = strong_pairs.get((face, speaker), 0)
            if score == 0:
                score = weak_pairs.get((face, speaker), 0)
            if score < min_cooccur:
                continue
            # Prefer higher co-occurrence, then the earlier-appearing face.
            candidate = (score, -face_first_ts[face])
            if best_score is None or candidate > best_score:
                best_score = candidate
                best_face = face
        if best_face is not None:
            assignment[speaker] = best_face
    return assignment
