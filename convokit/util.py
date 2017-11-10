import urllib.request
import shutil
import os
import pkg_resources

# returns a path to the dataset file
def download(name, verbose=True):
    """Use this to download (or use saved) convokit data by name.

    :param name: Which item to download. Currently supported:

        - "wiki-corpus": Wikipedia Talk Page Conversations Corpus (see
            http://www.cs.cornell.edu/~cristian/Echoes_of_power.html)
        - "supreme-corpus": Supreme Court Dialogs Corpus (see
            http://www.cs.cornell.edu/~cristian/Echoes_of_power.html)
        - "parliament-corpus": UK Parliament Question-Answer Corpus (see
            http://www.cs.cornell.edu/~cristian/Asking_too_much.html)

    :return: The path to the downloaded item.
    """
    DatasetURLs = {
        "parliament-corpus": "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/parliament-corpus/full.json",
        "supreme-corpus": "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/supreme-corpus/full.json",
        "tennis-corpus": "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/tennis-corpus/full.json",
        "wiki-corpus": "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/wiki-corpus/full.json",
        "parliament-motifs": [
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/parliament-corpus/parliament-motifs/answer_arcs.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/parliament-corpus/parliament-motifs/question_arcs.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/parliament-corpus/parliament-motifs/question_fits.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/parliament-corpus/parliament-motifs/question_fits.json.super",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/parliament-corpus/parliament-motifs/question_supersets_arcset_to_super.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/parliament-corpus/parliament-motifs/question_supersets_sets.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/parliament-corpus/parliament-motifs/question_tree_arc_set_counts.tsv",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/parliament-corpus/parliament-motifs/question_tree_downlinks.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/parliament-corpus/parliament-motifs/question_tree_edges.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/parliament-corpus/parliament-motifs/question_tree_uplinks.json"

        ],
        "supreme-motifs": [
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/supreme-corpus/supreme-motifs/answer_arcs.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/supreme-corpus/supreme-motifs/question_arcs.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/supreme-corpus/supreme-motifs/question_fits.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/supreme-corpus/supreme-motifs/question_fits.json.super",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/supreme-corpus/supreme-motifs/question_supersets_arcset_to_super.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/supreme-corpus/supreme-motifs/question_supersets_sets.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/supreme-corpus/supreme-motifs/question_tree_arc_set_counts.tsv",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/supreme-corpus/supreme-motifs/question_tree_downlinks.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/supreme-corpus/supreme-motifs/question_tree_edges.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/supreme-corpus/supreme-motifs/question_tree_uplinks.json"

        ],
        "tennis-motifs": [
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/tennis-corpus/tennis-motifs/answer_arcs.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/tennis-corpus/tennis-motifs/question_arcs.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/tennis-corpus/tennis-motifs/question_fits.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/tennis-corpus/tennis-motifs/question_fits.json.super",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/tennis-corpus/tennis-motifs/question_supersets_arcset_to_super.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/tennis-corpus/tennis-motifs/question_supersets_sets.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/tennis-corpus/tennis-motifs/question_tree_arc_set_counts.tsv",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/tennis-corpus/tennis-motifs/question_tree_downlinks.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/tennis-corpus/tennis-motifs/question_tree_edges.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/tennis-corpus/tennis-motifs/question_tree_uplinks.json"

        ],
        "wiki-motifs": [
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/wiki-corpus/wiki-motifs/answer_arcs.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/wiki-corpus/wiki-motifs/question_arcs.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/wiki-corpus/wiki-motifs/question_fits.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/wiki-corpus/wiki-motifs/question_fits.json.super",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/wiki-corpus/wiki-motifs/question_supersets_arcset_to_super.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/wiki-corpus/wiki-motifs/question_supersets_sets.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/wiki-corpus/wiki-motifs/question_tree_arc_set_counts.tsv",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/wiki-corpus/wiki-motifs/question_tree_downlinks.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/wiki-corpus/wiki-motifs/question_tree_edges.json",
            "http://zissou.infosci.cornell.edu/socialkit/" + \
            "datasets/wiki-corpus/wiki-motifs/question_tree_uplinks.json"

        ]

    }
    name = name.lower()
    data_dir = pkg_resources.resource_filename("convokit", "")
    parent_dir = os.path.join(data_dir, "downloads")
    dataset_path = os.path.join(data_dir, "downloads", name)
    if not os.path.exists(os.path.dirname(dataset_path)):
        os.makedirs(os.path.dirname(dataset_path))

    downloadeds_path = os.path.join(data_dir, "downloads", "downloaded.txt")
    if not os.path.isfile(downloadeds_path):
        open(downloadeds_path, "w").close()
    with open(downloadeds_path, "r") as f:
        downloaded = f.read().splitlines()

    if name not in downloaded:
        if "motifs" in name:
            for url in DatasetURLs[name]:
                full_name = name + url[url.rfind('/'):]
                if full_name not in downloaded:
                    motif_file_path = dataset_path + url[url.rfind('/'):]
                    if not os.path.exists(os.path.dirname(motif_file_path)):
                        os.makedirs(os.path.dirname(motif_file_path))
                    download_helper(motif_file_path, url, verbose, full_name, downloadeds_path)
            parent_dir = os.path.join(parent_dir, name)
        else:
            url = DatasetURLs[name]
            download_helper(dataset_path, url, verbose, name, downloadeds_path)
    return parent_dir

def download_helper(dataset_path, url, verbose, name, downloadeds_path):
    with urllib.request.urlopen(url) as response, \
                open(dataset_path, "wb") as out_file:
            if verbose:
                l = float(response.info()["Content-Length"])
                length = str(round(l / 1e6, 1)) + "MB" \
                        if l > 1e6 else \
                        str(round(l / 1e3, 1)) + "KB"
                print("Downloading", name, "from", url,
                        "(" + length + ")...", end=" ", flush=True)
            shutil.copyfileobj(response, out_file)
            if verbose:
                print("Done")
            with open(downloadeds_path, "a") as f:
                f.write(name + "\n")
