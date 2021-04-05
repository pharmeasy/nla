import os
import transly.pronunciation as tp


filepath = os.path.dirname(os.path.abspath(__file__))

pronunciation = tp.load_model(model_path="cmu")
wordgen = tp.load_model(filepath + "/models/homophones/", "model.h5")


def genome(query, beamwidth=50):
    query = query.upper()

    result = []
    splits = [4]

    if len(query) < 10:
        result += wordgen.beamsearch(
            pronunciation.infer(query, " "), mode="", beam_width=beamwidth // 3
        )
        splits += [3]

    for nsplit in splits:
        beamw = 3 if len(splits) == 2 else 1

        qsplit = (
            "".join(
                [c + " " if (i + 1) % nsplit == 0 else c for i, c in enumerate(query)]
            )
            .strip()
            .split()
        )

        if len(qsplit[-1]) < 3:
            qsplit[-2] = qsplit[-2] + qsplit[-1]
            qsplit = qsplit[:-1]

        z = [
            wordgen.beamsearch(
                pronunciation.infer(word, " "), mode="", beam_width=beamwidth // beamw
            )
            for word in qsplit
        ]
        result += ["".join(i) for i in zip(*z)]
    return list(set(result))


if __name__ == "__main__":
    print(genome("CARROM", beamwidth=20))
