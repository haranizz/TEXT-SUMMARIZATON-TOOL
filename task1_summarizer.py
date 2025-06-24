import nltk
nltk.download('punkt')

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

if __name__ == "__main__":
    input_text = """
    Artificial Intelligence (AI) is one of the most transformative technologies of the 21st century.
    It has found applications in almost every industry—from healthcare and finance to transportation and education.
    AI can analyze vast amounts of data at speeds unimaginable for humans, allowing businesses to make more informed decisions.
    In healthcare, AI is used to detect diseases early, assist in surgeries, and personalize treatment plans.
    In transportation, self-driving cars are becoming more reliable due to advances in AI algorithms.
    However, with its rise, ethical concerns have emerged, such as data privacy, algorithmic bias, and job displacement.
    Policymakers and researchers are actively working to create regulations and frameworks to ensure the responsible development and use of AI.
    The future of AI holds great promise, but it must be handled with care to ensure that its benefits are maximized while minimizing its risks.
    """

    print("Original word count:", len(input_text.split()))
    print("\nSummary:\n", summarize_text(input_text, sentence_count=3))
