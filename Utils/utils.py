import nltk
nltk.download('punkt_tab', download_dir='nltk_data')
nltk.download('punkt', download_dir='nltk_data')
nltk.data.path.append('nltk_data')



def split_paragraph(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return sentences

def reformat_to_string(corpus: list[dict]):
    reformatted_corpus = []
    for i in range(0, len(corpus)):
        doc_dictionary = corpus[i]
        title = doc_dictionary['TITLE']
        abstracts = doc_dictionary["ABSTRACT"]
        concatenated_abstracts = "".join(map(str, abstracts))
        reformatted_doc = f"Title:{title}\nAbstract:{concatenated_abstracts}"
        reformatted_corpus.append(reformatted_doc)
    
    return reformatted_corpus