import xml.etree.ElementTree as ET

from matchmaker_eval.score_input import prepared_score_path


def test_preserves_normal_score_and_source_bytes(tmp_path):
    path = tmp_path / 'score.musicxml'
    source = b'<score-partwise><part id="P1"><note><voice>2</voice></note></part></score-partwise>'
    path.write_bytes(source)
    with prepared_score_path(path) as (prepared, changes):
        assert prepared == path
        assert changes == []
    assert path.read_bytes() == source


def test_remaps_large_voice_labels_without_merging_voices(tmp_path):
    path = tmp_path / 'score.musicxml'
    source = b'''<score-partwise><part id="P1"><measure>
      <note id="a"><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
      <note id="b"><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration><voice>140142961047633</voice></note>
      <note id="c"><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><voice>140142961047633</voice></note>
      <note id="d"><pitch><step>F</step><octave>4</octave></pitch><duration>4</duration><voice>140142952291521</voice></note>
      </measure></part></score-partwise>'''
    path.write_bytes(source)
    with prepared_score_path(path) as (prepared, changes):
        assert prepared != path
        root = ET.parse(prepared)
        assert [e.text for e in root.findall('.//voice')] == ['1', '2', '2', '3']
        assert changes == [{'part': 'P1', 'voice_ids': {'140142961047633': 2, '140142952291521': 3}}]
        old = ET.fromstring(source)
        for e in old.findall('.//voice'):
            e.text = 'ignored'
        for e in root.findall('.//voice'):
            e.text = 'ignored'
        assert ET.tostring(root.getroot()) == ET.tostring(old)
    assert not prepared.exists()
    assert path.read_bytes() == source


def test_voice_labels_are_separate_between_parts(tmp_path):
    path = tmp_path / 'score.xml'
    path.write_text('<score-partwise><part id="A"><note><voice>9999999999</voice></note></part>'
                    '<part id="B"><note><voice>1</voice></note><note><voice>9999999999</voice></note></part></score-partwise>')
    with prepared_score_path(path) as (prepared, changes):
        assert [e.text for e in ET.parse(prepared).findall('.//voice')] == ['1', '1', '2']
