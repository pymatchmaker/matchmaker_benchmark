# The example piece — attribution and licence

`resources/` holds one performance, committed directly to this repository so
that the `example` fold — the smoke test every pull request runs — needs no
dataset download. It is **not** benchmark code, and it is **not** covered by
this repository's Apache-2.0 licence.

## What is here

J. S. Bach, Fugue in A-flat major, BWV 858 (*Das wohltemperierte Klavier* I),
performance `VuV01M`.

| File | Content | Origin |
| --- | --- | --- |
| `ex_score.musicxml` | Score | (n)ASAP |
| `ex_score.mid` | Score MIDI | (n)ASAP |
| `ex_VuV01M.mid` | Performance MIDI | (n)ASAP |
| `ex_VuV01M.wav` | Performance audio | MAESTRO v2.0.0 |
| `ex_VuV01M.match` | Note-level alignment | (n)ASAP |
| `*_annotations.txt` | Beat and note annotations | (n)ASAP |

The same piece appears in the evaluation data as
`asap/Bach/Fugue/bwv_858/VuV01M`; the copies here are the same material under
shorter names.

## Source

Symbolic data (MIDI, MusicXML score, note alignment) from the **(n)ASAP
dataset**: <https://github.com/CPJKU/asap-dataset>

Audio from the **MAESTRO dataset v2.0.0**:
<https://magenta.tensorflow.org/datasets/maestro>

ASAP does not distribute audio itself; its README directs users to MAESTRO and
`initialize_dataset.py` to pair the recordings with the ASAP performances.

## Licence

Both sources are released under
**[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)** — full
text in [LICENSE-CC-BY-NC-SA-4.0.md](LICENSE-CC-BY-NC-SA-4.0.md).

Anyone using these files must:

- **credit** both (n)ASAP and MAESTRO (citations below);
- not use them **commercially**;
- distribute any derivative under the **same licence**.

Share-alike applies to this folder and to anything derived from it. The rest of
the repository — the evaluation code, the fold definitions, the leaderboard —
stays under Apache-2.0; see the note in the root [LICENSE](../LICENSE).

## Citation

```bibtex
@article{Peter-2023,
 title = {Automatic Note-Level Score-to-Performance Alignments in the ASAP Dataset},
 author = {Peter, Silvan David and Cancino-Chacón, Carlos Eduardo and Foscarin, Francesco and McLeod, Andrew Philip and Henkel, Florian and Karystinaios, Emmanouil and Widmer, Gerhard},
 doi = {10.5334/tismir.149},
 journal = {Transactions of the International Society for Music Information Retrieval {(TISMIR)}},
 year = {2023}
}

@inproceedings{asap-dataset,
  title     = {{ASAP}: a dataset of aligned scores and performances for piano transcription},
  author    = {Foscarin, Francesco and McLeod, Andrew and Rigaux, Philippe and
               Jacquemard, Florent and Sakai, Masahiko},
  booktitle = {International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2020}
}

@inproceedings{hawthorne2019maestro,
  title     = {Enabling Factorized Piano Music Modeling and Generation with the
               {MAESTRO} Dataset},
  author    = {Hawthorne, Curtis and Stasyuk, Andriy and Roberts, Adam and
               Simon, Ian and Huang, Cheng-Zhi Anna and Dieleman, Sander and
               Elsen, Erich and Engel, Jesse and Eck, Douglas},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019}
}
```
