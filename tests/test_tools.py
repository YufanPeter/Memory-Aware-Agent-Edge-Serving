from edgeserve.delays import DelaySampler
from edgeserve.eval import exact_match, f1_score
from edgeserve.tools import ToolBox


def test_find_title(corpus):
    assert corpus.find_title("Scott Derrickson") == "Scott Derrickson"
    assert corpus.find_title("scott derrickson!") == "Scott Derrickson"
    assert corpus.find_title("Ed Wood") == "Ed Wood"                       # exact beats the "(film)" page
    assert corpus.find_title("Doctor Strange") == "Doctor Strange (2016 film)"
    assert corpus.find_title("Scot Derickson") == "Scott Derrickson"        # fuzzy
    assert corpus.find_title("Completely Unknown Thing") is None


def test_search_provenance(corpus, item):
    tb = ToolBox(corpus, item.sf_set, search_sentences=2)
    obs = tb.search("Scott Derrickson")
    assert obs.text == "[Scott Derrickson] Scott Derrickson (born July 16, 1966) is an American director. He lives in Los Angeles, California."
    assert [k for k, _ in obs.pieces] == ["obs_meta", "obs_sf", "obs_other"]
    assert obs.sf_hits == {("Scott Derrickson", 0)}


def test_search_not_found_and_lookup(corpus, item):
    tb = ToolBox(corpus, item.sf_set)
    assert tb.lookup("born").text.startswith("No paragraph searched")
    assert tb.search("Zzzz Qqqq").text.startswith("Could not find [Zzzz Qqqq]. Similar:")
    tb.search("Ed Wood")
    first = tb.lookup("born")
    assert first.text == "(Result 1 / 1) He was born in Poughkeepsie, New York."
    assert tb.lookup("born").text == "No more results."


def test_search_miss_returns_closest_text(corpus, item):
    corpus.text_search = lambda q, n: ["Ed Wood", "Scott Derrickson", "Ed Wood (film)"]
    obs = ToolBox(corpus, item.sf_set, search_sentences=1).search("filmmaker from Poughkeepsie")
    assert obs.text == ("Could not find [filmmaker from Poughkeepsie]. Closest result: "
                        "[Ed Wood] Edward Davis Wood Jr. was an American filmmaker.\n"
                        "Similar: ['Scott Derrickson', 'Ed Wood (film)'].")
    assert obs.sf_hits == {("Ed Wood", 0)}
    long = ToolBox(corpus, item.sf_set, search_sentences=0, neighbors=1).search("filmmaker")
    assert long.text.count("[") == 4   # two paragraphs shown, one suggestion left
    tb = ToolBox(corpus, item.sf_set)
    tb.search("filmmaker from Poughkeepsie")
    assert tb.lookup("born").text == "(Result 1 / 1) He was born in Poughkeepsie, New York."


def test_paragraphs_whole(corpus, item):
    obs = ToolBox(corpus, item.sf_set).paragraphs(["Ed Wood"])
    assert obs.text == "[Ed Wood] Edward Davis Wood Jr. was an American filmmaker. He was born in Poughkeepsie, New York.\n"


def test_delays_deterministic_and_order_free():
    d = DelaySampler({"type": "lognormal", "median": 1.0, "sigma": 0.5}, {"lookup": {"type": "fixed", "seconds": 0.1}})
    a = [d.sample("search", 0, "t1", s) for s in range(5)]
    _ = d.sample("search", 0, "t2", 0)
    b = [d.sample("search", 0, "t1", s) for s in reversed(range(5))][::-1]
    assert a == b
    assert d.sample("lookup", 0, "t1", 0) == 0.1
    assert DelaySampler(scale=2.0).sample("search", 0, "t", 0) == 0.0


def test_em_f1():
    assert exact_match("The Beatles", "beatles") == 1.0
    assert f1_score("yes", "no") == 0.0
    assert round(f1_score("Los Angeles California", "Los Angeles"), 3) == 0.8
