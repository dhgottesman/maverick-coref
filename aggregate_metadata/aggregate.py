import os
import ujson as json
from tqdm import tqdm

import sys
import csv
import glob
import argparse

from utils import *

from concurrent.futures import ProcessPoolExecutor, as_completed

def _fetch(args):
    file_path, doc_id, byte_start, byte_end = args
    with open(file_path, 'r') as f:
        f.seek(byte_start)
        line = f.read(byte_end - byte_start)
        data = json.loads(line)
        assert doc_id == data["id"]
        return doc_id, data

def stream_ndjson_with_offsets(file_path, doc_ids=None, window_size=1000):
    """
    If doc_ids is provided and the offset map exists, 
    return only those docs in the given order.
    """
    offset_map_path = file_path.replace(".json", "_index.json")
    id_offset_map = {}

    if os.path.exists(offset_map_path):
        if doc_ids is None:
            raise Exception("Must specify doc_ids.")

        with open(offset_map_path, 'r') as f:
            id_offset_map = json.load(f)
        id_offset_map = {k: tuple(v) for k, v in id_offset_map.items()}

        for i in range(0, len(doc_ids), window_size):
            window = doc_ids[i:i + window_size]
            tasks = [
                (file_path, doc_id, *id_offset_map[doc_id])
                for doc_id in window if doc_id in id_offset_map
            ]

            with ProcessPoolExecutor(max_workers=25) as executor:
                futures = {executor.submit(_fetch, task): task[1] for task in tasks}
                results = {future.result()[0]: future.result()[1] for future in as_completed(futures)}

            for doc_id in window:
                if doc_id in results:
                    yield results[doc_id]

    else:
        id_offset_map = {}
        with open(file_path, 'r') as f:
            while True:
                byte_start = f.tell()
                line = f.readline()
                if not line:
                    break
                byte_end = f.tell()

                data = json.loads(line)
                doc_id = data.get("id")
                if doc_id:
                    id_offset_map[doc_id] = (byte_start, byte_end)
                    yield data

        with open(offset_map_path, 'w') as out:
            json.dump({k: list(v) for k, v in id_offset_map.items()}, out)

def process_doc(doc, skip_coref=False):
    hyperlinks = normalize_hyperlinks(doc.get("hyperlinks_clean", []))
    coref = doc.get("coref", [])
    entity_linking = normalize_entity_linking(doc.get("entity_linking", []))

    if not skip_coref:
        enriched_clusters = enrich_coref_clusters(coref, entity_linking + hyperlinks)
        enriched_clusters = {
            cid: spans for cid, spans in enriched_clusters.items()
            if any(span.get("entities") or span.get("links") for span in spans)
        }

        entity_scores, enriched_clusters = score_entities_by_subject_likelihood(enriched_clusters)
        filtered_enriched_clusters = {}
        for cid, spans in enriched_clusters.items():
            scores = [v for span in spans for v in span.get("score", {}).values()]
            if scores and max(scores) >= 0.3:
                filtered_enriched_clusters[cid] = spans
    else:
        print(f"Skipping coref for doc {doc['id']}")
        filtered_enriched_clusters, entity_scores = {}, {}

    return aggregate_mentions(hyperlinks, entity_linking, filtered_enriched_clusters, entity_scores)

def enrich_csv_with_entities(csv_path, ndjson_path, output_path, doc_ids, headers):
    # Determine where to resume
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', newline='') as existing_out:
            reader = csv.DictReader(existing_out, fieldnames=headers)
            for row in reader:
                processed_ids.add(row["id"])

    with open(csv_path, 'r', newline='') as in_f, \
         open(output_path, 'a', newline='') as out_f:

        reader = csv.DictReader(in_f, fieldnames=headers)
        writer = csv.DictWriter(out_f, fieldnames=headers)

        # Create generator over NDJSON docs (streaming)
        if processed_ids:
            for i, doc_id in enumerate(doc_ids):
                if doc_id not in processed_ids:
                    doc_ids = doc_ids[i:]
                    break
            else:
                doc_ids = []
        print(f"STARTING {doc_ids[0]}")
        ndjson_iter = stream_ndjson_with_offsets(ndjson_path, doc_ids)

        # Buffered NDJSON docs by ID
        doc_buffer = {}
        ndjson_finished = False

        for row in tqdm(reader):
            row_id = row["id"]

            # Skip already processed rows
            if row_id in processed_ids:
                continue

            # Fill buffer until this ID is available or NDJSON ends
            while row_id not in doc_buffer and not ndjson_finished:
                try:
                    doc = next(ndjson_iter)
                    doc_buffer[doc["id"]] = doc
                except StopIteration:
                    ndjson_finished = True
                    break

            doc = doc_buffer.pop(row_id, None)

            if doc:
                row["entities"] = json.dumps(process_doc(doc, row_id == '44471088' or row_id == "62180015" or row_id == "67215817"), ensure_ascii=False)
            else:
                raise Exception(f"No doc found for id {row_id}")
                
            try:
                writer.writerow(row)
                out_f.flush()
                os.fsync(out_f.fileno())
            except Exception as e:
                print(f"❌ Error during write: {e}")
                raise(e)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple script with one index argument.")
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Index of the item to process"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Increase CSV field size limit
    csv.field_size_limit(sys.maxsize)

    headers = ['start', 'end', 'id', 'src', 'loc', 'title', 'entities', 'offsets']

    csv_files = glob.glob(
        "/home/morg/students/gottesman3/knowledge-analysis-suite/dolma/python/final_tokenizations_with_offsets/no_special/*.csv"
    )

    src = set()
    doc_ids = []
    with open(csv_files[args.index], 'r', newline='') as f:
        reader = csv.DictReader(f, fieldnames=headers)
        for line_number, row in tqdm(enumerate(reader)):
            src.add(row["src"])
            doc_ids.append(row["id"])

    assert(len(src) == 1)
    
    csv_file = csv_files[args.index]
    ndjson_file = list(src)[0]
    output_file = csv_file.replace(".csv", "_new_2.csv")

    enrich_csv_with_entities(csv_file, ndjson_file, output_file, doc_ids, headers)

