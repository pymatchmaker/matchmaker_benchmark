"""Compatibility handling for MusicXML voice identifiers, not note content."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET


@contextmanager
def prepared_score_path(path):
    """Renumber oversized integer voice labels in a temporary MusicXML copy.

    Some exported scores use memory-address-sized voice identifiers. Partitura
    stores voices as int32; NumPy 2 rejects these values where NumPy 1 silently
    wrapped them. A voice identifier names a group, so a one-to-one replacement
    preserves voice membership. Existing int32 labels and every other score
    field remain unchanged. Original dataset files are never rewritten.
    """
    path = Path(path)
    if path.suffix.lower() not in ('.xml', '.musicxml'):
        yield path, []
        return
    tree = ET.parse(path)
    changes = []
    for part in tree.getroot().iter():
        if part.tag.rsplit('}', 1)[-1] != 'part':
            continue
        voices = [e for e in part.iter() if e.tag.rsplit('}', 1)[-1] == 'voice']
        values = [int(e.text) for e in voices]
        used = {v for v in values if -(2**31) <= v < 2**31}
        replacement = {}
        candidate = 1
        for element, value in zip(voices, values):
            if -(2**31) <= value < 2**31:
                continue
            if value not in replacement:
                while candidate in used:
                    candidate += 1
                replacement[value] = candidate
                used.add(candidate)
            element.text = str(replacement[value])
        if replacement:
            changes.append({'part': part.get('id'),
                            'voice_ids': {str(k): v for k, v in replacement.items()}})
    if not changes:
        yield path, []
        return
    with TemporaryDirectory(prefix='matchmaker-score-') as folder:
        normalized = Path(folder) / path.name
        tree.write(normalized, encoding='utf-8', xml_declaration=True)
        yield normalized, changes
