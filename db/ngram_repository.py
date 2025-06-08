from sqlalchemy.orm import Session
from db.ngram_index import NGramIndex as NGramIndexDB
import json
from ongoing_process_state.n_gram_index import NGramIndex
import random
from typing import List

def save_n_gram_index_to_db(n_gram_index: NGramIndex, process_id: str, db: Session):
    for prefix, marking_ids in n_gram_index.markings.items():
        prefix_str = ",".join(prefix)
        element_sets = [
            list(n_gram_index.graph.markings[marking_id])
            for marking_id in marking_ids
        ]
        marking_str = json.dumps(element_sets)

        entry = NGramIndexDB(
            process_id=process_id,
            prefix=prefix_str,
            marking=marking_str,
        )
        db.merge(entry)
    db.commit()

def ngram_index_exists_in_db(process_id: str, db: Session) -> bool:
    return db.query(NGramIndexDB).filter(NGramIndexDB.process_id == process_id).first() is not None

def get_best_marking_state_for_ngram_from_db(n_gram: List[str], process_id: str, db: Session) -> List[str]:
    final_marking = []

    for k in range(1, len(n_gram) + 1):
        suffix = ",".join(n_gram[-k:])
        result = db.query(NGramIndexDB).filter(
            NGramIndexDB.process_id == process_id,
            NGramIndexDB.prefix == suffix
        ).first()

        if result:
            markings = json.loads(result.marking)
            if len(markings) == 1:
                return markings[0]
            elif len(markings) > 1:
                final_marking = random.choice(markings)

    return final_marking
