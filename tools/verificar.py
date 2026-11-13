"""Check Barenn executables on Optdigits and a hand-calculated fixture."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
import unicodedata


@dataclass
class Results:
    train_size: int
    epochs: dict
    hits: int
    test_size: int
    test_loss: float
    predictions: list
    confusions: list


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_output(stdout):
    text = unicodedata.normalize("NFKD", stdout)
    text = "".join(char for char in text if not unicodedata.combining(char))
    lines = [line.strip() for line in text.splitlines()]

    def unique(pattern):
        matches = [m.groups() for line in lines if (m := re.fullmatch(pattern, line, re.I))]
        check(len(matches) == 1, f"esperada uma linha para {pattern!r}: {len(matches)}")
        return matches[0]

    train_size, = unique(r"TREINO\s*\|\s*(\d+)\s+imagens")
    hits, test_size = unique(r"Acuracia\s+\S+%\s*\|\s*(\d+)\s+acertos\s+em\s+(\d+)\s+imagens")
    test_loss, = unique(r"Perda\s+media\s+(\S+)")
    epochs = {}
    section = None
    predictions = []
    confusions = []
    saw_confusions = False
    no_confusions = False
    for line in lines:
        match = re.fullmatch(r"Epoca\s+(\d+)/(\d+)\s*\|\s*perda media\s+(\S+)", line, re.I)
        if match:
            epoch, total, loss = match.groups()
            key = (int(epoch), int(total))
            check(key not in epochs, f"época repetida: {epoch}")
            epochs[key] = float(loss)
        elif line.upper() == "ALGUMAS INFERENCIAS":
            section = predictions
        elif line.upper() == "CONFUSOES MAIS FREQUENTES":
            section = confusions
            saw_confusions = True
        elif section is predictions and re.fullmatch(r"\d+\s+\d+", line):
            predictions.append(tuple(map(int, line.split())))
        elif section is confusions and re.fullmatch(r"\d+\s+\d+\s+\d+", line):
            confusions.append(tuple(map(int, line.split())))
        elif section is confusions and line.lower().startswith("nenhuma "):
            no_confusions = True

    check(list(epochs) == [(i, 51) for i in range(1, 52, 5)], "lista de épocas inesperada")
    check(predictions and saw_confusions and bool(confusions) != no_confusions,
          "exemplos de inferência ou confusões ausentes/inconsistentes")
    test_loss = float(test_loss)
    check(all(math.isfinite(value) and value >= 0 for value in [*epochs.values(), test_loss]),
          "perda negativa, infinita ou NaN")
    return Results(int(train_size), epochs, int(hits), int(test_size), test_loss,
                   predictions, confusions)


def run(executable, train, test, timeout):
    result = subprocess.run(
        [str(executable), str(train), str(test)],
        cwd=train.parent, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
    )
    check(result.returncode == 0, f"{executable.name} saiu com código {result.returncode}: "
          f"{result.stderr.strip() or result.stdout.strip()}")
    try:
        return read_output(result.stdout)
    except ValueError as error:
        raise ValueError(f"{executable.name} ({train.name}): {error}\n{result.stdout}") from error


def check_full(reference, candidate, atol):
    for name in ("train_size", "test_size", "hits", "predictions", "confusions"):
        expected = getattr(reference, name)
        actual = getattr(candidate, name)
        check(actual == expected, f"{name}: legado={expected!r}, candidato={actual!r}")
    check((candidate.train_size, candidate.test_size) == (3823, 1797),
          "contagens esperadas para optdigits.tra e optdigits.tes: 3823 e 1797")
    for epoch, ref in reference.epochs.items():
        actual = candidate.epochs[epoch]
        check(math.isclose(ref, actual, abs_tol=atol, rel_tol=0), f"época {epoch[0]}: {ref} vs {actual}")
    check(math.isclose(reference.test_loss, candidate.test_loss, abs_tol=atol, rel_tol=0),
          f"perda de teste: {reference.test_loss} vs {candidate.test_loss}")


def zero_fixture(directory):
    csv = "0," * 64 + "2\n" + "0," * 64 + "7\n"
    train = directory / "zeros-train.csv"
    test = directory / "zeros-test.csv"
    train.write_text(csv, encoding="ascii")
    test.write_text(csv, encoding="ascii")
    return train, test


def zero_reference():
    scores = [0.0, 0.0]

    def squared_loss(label):
        return 0.5 * sum((score - (index == label)) ** 2 for index, score in enumerate(scores))

    epochs = {}
    for epoch in range(1, 52):
        loss = 0.0
        for label in (0, 1):
            loss += squared_loss(label)
            for index, score in enumerate(scores):
                scores[index] -= 0.02 * (score - (index == label))
        if (epoch - 1) % 5 == 0:
            epochs[(epoch, 51)] = loss / 2
    return Results(2, epochs, 1, 2, (squared_loss(0) + squared_loss(1)) / 2,
                   [(2, 7), (7, 7)], [(2, 7, 1)])


def check_zero(actual):
    expected = zero_reference()
    for name in ("train_size", "test_size", "hits", "predictions", "confusions"):
        check(getattr(actual, name) == getattr(expected, name),
              f"fixture de zeros: {name}={getattr(actual, name)!r}, esperado {getattr(expected, name)!r}")
    # The executable rounds losses to four decimal places.
    tolerance = 0.000051
    for epoch, loss in expected.epochs.items():
        check(math.isclose(actual.epochs[epoch], loss, abs_tol=tolerance, rel_tol=0),
              f"fixture de zeros, época {epoch[0]}: {actual.epochs[epoch]} vs {loss:.8f}")
    check(math.isclose(actual.test_loss, expected.test_loss, abs_tol=tolerance, rel_tol=0),
          f"fixture de zeros, perda de teste: {actual.test_loss} vs {expected.test_loss:.8f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="pasta com optdigits.tra/tes")
    parser.add_argument("--atol", type=float, default=1e-4, help="tolerância absoluta das perdas Optdigits")
    parser.add_argument("--timeout", type=float, default=300, help="limite em segundos por execução")
    args = parser.parse_args()

    try:
        check(math.isfinite(args.atol) and args.atol >= 0, "--atol deve ser finito e não negativo")
        check(math.isfinite(args.timeout) and args.timeout > 0, "--timeout deve ser finito e positivo")
        root = args.repo.resolve()
        train, test = root / "optdigits.tra", root / "optdigits.tes"
        legacy, candidate = args.legacy.resolve(), args.candidate.resolve()
        reference = run(legacy, train, test, args.timeout)
        actual = run(candidate, train, test, args.timeout)
        check_full(reference, actual, args.atol)
        print(f"Optdigits: {actual.hits}/{actual.test_size} acertos, perda {actual.test_loss:.4f}.")
        with TemporaryDirectory(prefix="aps2-zero-") as directory:
            train, test = zero_fixture(Path(directory))
            check_zero(run(legacy, train, test, args.timeout))
            check_zero(run(candidate, train, test, args.timeout))
        print("Fixture de zeros: 1/2 acertos, perda 0.2541, confusão 2 -> 7 (1).")
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Falha: {error}\n")


if __name__ == "__main__":
    main()