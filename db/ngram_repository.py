from sqlalchemy.orm import Session
from db.ngram_index import NGramIndex as NGramIndexDB
import json
from ongoing_process_state.n_gram_index import NGramIndex
from ongoing_process_state.reachability_graph import ReachabilityGraph

def save_n_gram_index_to_db(n_gram_index: NGramIndex, process_id: str, db: Session):
    for prefix, marking_ids in n_gram_index.markings.items():
        prefix_str = ",".join(prefix)
        marking_str = json.dumps(list(marking_ids))

        entry = NGramIndexDB(
            process_id=process_id,
            prefix=prefix_str,
            marking=marking_str,
        )
        db.merge(entry)
    db.commit()

def load_n_gram_index_from_db(process_id: str, reachability_graph: ReachabilityGraph, db: Session) -> NGramIndex:
    n_gram_index = NGramIndex(graph=reachability_graph, n_gram_size_limit=20)
    entries = db.query(NGramIndexDB).filter(NGramIndexDB.process_id == process_id).all()

    if not entries:
        raise RuntimeError(f"No n-gram index entries found in DB for process_id={process_id}")

    for entry in entries:
        prefix = entry.prefix.split(",")
        marking_ids = json.loads(entry.marking)
        for marking in marking_ids:
            n_gram_index.add_association(prefix, marking)

    return n_gram_index
