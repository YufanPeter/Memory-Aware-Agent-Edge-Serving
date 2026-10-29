import pytest

from edgeserve.corpus import Corpus
from edgeserve.data import HotpotItem

DOCS = {
    "Scott Derrickson": ["Scott Derrickson (born July 16, 1966) is an American director.",
                         " He lives in Los Angeles, California.",
                         " He is best known for directing horror films."],
    "Ed Wood": ["Edward Davis Wood Jr. was an American filmmaker.",
                " He was born in Poughkeepsie, New York."],
    "Ed Wood (film)": ["Ed Wood is a 1994 American biographical film directed by Tim Burton."],
    "Doctor Strange (2016 film)": ["Doctor Strange is a 2016 superhero film directed by Scott Derrickson."],
}


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(dict(DOCS))


@pytest.fixture
def item() -> HotpotItem:
    return HotpotItem(
        id="q1", question="Were Scott Derrickson and Ed Wood of the same nationality?", answer="yes",
        qtype="comparison",
        titles=["Scott Derrickson", "Ed Wood", "Ed Wood (film)"],
        paragraphs=[DOCS["Scott Derrickson"], DOCS["Ed Wood"], DOCS["Ed Wood (film)"]],
        supporting_facts=[("Scott Derrickson", 0), ("Ed Wood", 0)],
    )
